import math
import gym
from gym import spaces
import numpy as np
from gym.envs.classic_control import rendering
import pyglet
pyglet.options["debug_gl"] = False

class CustomPong(gym.Env):
    """
    Description:
        Pong Game
    Observation:
        Type: Box(3)
        Num     Observation                                                                     Min                     Max
        0       Ball horizintal distance from paddle, as a proportion of game width             0                       1
        1       Paddle Height, as a proportion of game height                                   0                       1
        2       Ball height, as a proportion of game height                                     0                       1
        3       Ball speed proportion in x direction towards paddle                             0                       1
        4       Ball speed proportion in y                                                      0                       1
    Actions:
        Type: Discrete(3)
        Num   Action
        0     Move paddle down
        1     Move paddle up
        2     Do nothing
    Reward:
        Reward is 0.1 for hitting the ball the the paddle.
        Missing the ball incurs a loss proportional to the miss distance.
        Negative reward is incurred for every step to encorage attacking
    Starting State:
        Ball is initialized at the left side, with random x and y velocity
    Episode Termination:
        Each agent starts with 100 health.
        Missing incurs health loss proportional to the miss size.
        Game ends when one agent reaches 0 hp.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.ball_speed = 3.0
        self.paddle_speed = 3.0
        self.paddle_length = 50
        self.paddle_hit_reward = 0.1

        # Set size of the game field
        self.w = 512
        self.h = 256
        self.bounds = [0, self.w, 0, self.h]

        # To speed up hit detection, we don't check for collisions in this region
        self.center_bounds = [30, self.w-30, 30, self.h-30]
        
        # Setting meta-info for gym
        high = np.array(
            [
                1, # Ball distance
                1, # Paddle height
                1, # Ball height
                1, # Ball dx
                1, # Ball dy
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3) # Up, Down, Nothing
        self.observation_space = spaces.Box(np.zeros(high.shape[0]), high, dtype=np.float32)

        # Initialize state arrays once for efficiency
        self.state_r = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.state_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.reset()


        self.viewer = None

    def init_game(self):
        """ Initialize game components
        """

        # Create game components
        top_wall = PongBlock(l=0, r=self.w, t=self.h, b=self.h-10)
        bottom_wall = PongBlock(l=0, r=self.w, t=10, b=0)
        self.paddle_l = PongPaddle(l=0, r=10, t = (self.h//2) + (self.paddle_length//2), b= (self.h//2) - (self.paddle_length//2), ny=1, nx=-1, hit_reward_l=self.paddle_hit_reward, speed=self.paddle_speed, seed=self.seed)
        self.paddle_r = PongPaddle(l=self.w-10, r=self.w, t = (self.h//2) + (self.paddle_length//2), b= (self.h//2) - (self.paddle_length//2), ny=1, nx=-1, hit_reward_r=self.paddle_hit_reward, speed=self.paddle_speed, seed=self.seed)
        self.ball = PongBall(x=self.w//10, y = self.h//2, speed=self.ball_speed, dx=self.rng.rand()*0.75 + 0.25, dy=self.rng.rand()*0.75 + 0.25, seed=self.seed)

        # Setup objects to check for hit interactions
        self.interactions = [top_wall, bottom_wall, self.paddle_l, self.paddle_r]

        # Setup monitoring and trackers
        self.reward_r = 0
        self.reward_l = 0
        self.total_reward_r = 0
        self.total_reward_l = 0
        self.paddle_last_hit = None
        self.done = False
        self.last_failed_l = 100
        self.last_failed_r = 100
        self.hp_l = 100
        self.hp_r = 100

    def reset(self):
        """ Reset the system, initializing settings to base
        """
        seed = np.random.randint(10000)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.init_game()
        self.set_state()
        return self.state_l, self.state_r

    def set_state(self):
        """ Extract state information from game components.
        We modify the current state arrays for speed, so we don't need to reinitialize every step.
        """
        # Distance from ball to paddle (on right) as a proportion of screen width
        self.state_r[0] = (self.w - self.ball.current_x) / self.w    
        # Paddle (right) height as proportion of screen height
        self.state_r[1] = ((self.paddle_r.t + self.paddle_r.b) / 2) / self.h
        # Ball height as proportion of screen height
        self.state_r[2] = (self.ball.current_y) / self.h
        # Proportion of ball speed in x direction (towards right), rescalled to be between 0 and 1
        self.state_r[3] = (self.ball.dx + 1) / 2
        # Proportion of ball speed in y direction, rescalled to be between 0 and 1
        self.state_r[4] = (self.ball.dy + 1) / 2

        # Same as above for the left paddle, reversing values where appropriate
        self.state_l[0] = (self.ball.current_x) / self.w    
        self.state_l[1] = ((self.paddle_l.t + self.paddle_l.b) / 2) / self.h
        self.state_l[2] = (self.ball.current_y) / self.h
        self.state_l[3] = (self.ball.dx * -1 + 1) / 2
        self.state_l[4] = (self.ball.dy + 1) / 2

    def step(self, action):
        """ Move the game 1 step, taking the actions from the left and right players
        """

        # Turn the continuous output from the neural networks into discrete button presses.
        # We take the index of the max value to be the action,
        # [100, 200, 0] => 1
        # [200, 100, 0] => 0
        # [200, 200, 210] => 2
        action_l, action_r = action
        a_l = np.argmax(action_l)
        a_r = np.argmax(action_r)

        # Increment monitors (for the red flash on hit)
        self.last_failed_l += 1
        self.last_failed_r += 1
        
        # Calculate whether the ball is in the center of the area, to make calculations more efficient
        in_center = (self.center_bounds[0] < self.ball.current_x < self.center_bounds[1]) and (self.center_bounds[2] < self.ball.current_y < self.center_bounds[3])

        # Step ball, calculating hit rewards based on interactive objects
        (self.reward_l, self.reward_r), hit_obj = self.ball.step(self.interactions if not in_center else [])
        if hit_obj == self.paddle_l:
            self.paddle_last_hit = self.paddle_l
        elif hit_obj == self.paddle_r:
            self.paddle_last_hit = self.paddle_r

        # Paddles take steps based on passed actions
        # a = 0 means move up (1-0 = 1 = up)
        # a = 1 means no movement (1-1 = 0 = none)
        # a = 2 means move down (1-2 = -1 = down)
        dy = self.paddle_speed *(1 - a_r)
        self.paddle_r.step(self.bounds, dx=0, dy=dy)

        dy = self.paddle_speed *(1 - a_l)
        self.paddle_l.step(self.bounds, dx=0, dy=dy)

        # Handle hitting walls
        # Unline the top and bottem, there are no interactive objects here so we handle this manually
        # We set a hit to be crossing the x=4 and x = (w-4) marks. This is to stop the ball hitting the top of the paddle (easier if only interacts with the front)
        if self.ball.current_x >= self.w-6:
            # Calculate hp_loss as the distance from the impact to the paddle, as a proportion of screen height not taken up by the paddle
            hp_loss = (((min(abs(self.paddle_r.t - self.ball.current_y), abs(self.paddle_r.b - self.ball.current_y)) / (self.h - (self.paddle_r.t - self.paddle_r.b)))))
            # Set rewards and hp loss. Note we only set add to the other players reward if it was a direct attack
            self.reward_r -=  hp_loss
            if self.paddle_last_hit == self.paddle_l:
                self.reward_l += hp_loss
            self.hp_r -= hp_loss * 100

            # Make ball bounce by modifying directions
            self.ball.dx = -1 * self.ball.dx
            # Set back from the wall so it doesn't trigger next step
            self.ball.current_x = self.w - 11

            # Monitors
            self.paddle_last_hit = None
            self.last_failed_r = 0
        if self.ball.current_x <= 4:
            hp_loss = (((min(abs(self.paddle_l.t - self.ball.current_y), abs(self.paddle_l.b - self.ball.current_y)) / (self.h - (self.paddle_l.t - self.paddle_l.b)))))
            self.reward_l -=  hp_loss
            if self.paddle_last_hit == self.paddle_r:
                self.reward_r += hp_loss
            self.hp_l -= hp_loss * 100
            self.ball.dx = -1 * self.ball.dx
            self.ball.current_x = 11
            self.paddle_last_hit = None
            self.last_failed_l = 0
        
        # Game end condition
        if min(self.hp_l, self.hp_r) <= 0:
            self.done = True

        # Set the state so agents can act on it
        self.set_state()

        # Set rewards
        self.reward_l -= 0.0001
        self.reward_r -= 0.0001

        self.total_reward_l += self.reward_l
        self.total_reward_r += self.reward_r
        return (self.state_l, self.state_r), (self.reward_l, self.reward_r), self.done, None




    def render(self, mode="human"):
        """ Render current game state to screen
        """

        # Allow screen to be larger than game pixels
        screen_width = self.w * 2
        screen_height = self.h * 2
        world_width = self.w
        scale = screen_width / world_width

        # Set up geometry on the first render
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.score_label_l = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=(0+11)*scale,
                y=(self.h - 20)*scale,
                anchor_x="left",
                anchor_y="top",
                color=(1, 1, 255, 255),
            ))
            self.score_label_r = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=(self.w-11)*scale,
                y=(self.h - 20)*scale,
                anchor_x="right",
                anchor_y="top",
                color=(1, 1, 255, 255),
            ))
            self.hp_label_l = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=(0+11)*scale,
                y=(self.h - 40)*scale,
                anchor_x="left",
                anchor_y="top",
                color=(255, 1, 50, 255),
            ))
            self.hp_label_r = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=(self.w-11)*scale,
                y=(self.h - 40)*scale,
                anchor_x="right",
                anchor_y="top",
                color=(255, 1, 50, 255),
            ))
            self.viewer.add_geom(self.score_label_l)
            self.viewer.add_geom(self.score_label_r)
            self.viewer.add_geom(self.hp_label_l)
            self.viewer.add_geom(self.hp_label_r)

        # Set data
        for obj in self.interactions:
            rend = obj.render(self.viewer, scale)
        
        if self.last_failed_l < 10:
            self.paddle_l.geom.set_color(0.9, 0.0, 0.0)
        else:
            self.paddle_l.geom.set_color(0.0, 0.0, 0.0)
        if self.last_failed_r < 10:
            self.paddle_r.geom.set_color(0.9, 0.0, 0.0)
        else:
            self.paddle_r.geom.set_color(0.0, 0.0, 0.0)

        rend = self.ball.render(self.viewer, scale)
        self.score_label_l.label.text = f"{self.total_reward_l:.2f}"
        self.score_label_r.label.text = f"{self.total_reward_r:.2f}"
        self.hp_label_l.label.text = f"{self.hp_l:.0f}"
        self.hp_label_r.label.text = f"{self.hp_r:.0f}"

        r = self.viewer.render(return_rgb_array=mode == "rgb_array")

        return r

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class PongBall:
    def __init__(self, x=0, y=0, speed=1.0, dx=1, dy=1, seed=0):
        self.current_x = x
        self.current_y = y
        self.speed = speed
        # Normalize directions
        norm = np.linalg.norm([dx, dy])
        self.dx = dx / norm
        self.dy = dy / norm
        self.geom = None
        self.rng = np.random.RandomState(seed)

    
    def step(self, interactions):
        """ Step the ball. Step in each direction, and check for intersections.
        Use the normal of a hit to determine how direction changes.
        Return obj hit, and reward of the hit
        """
        self.current_x += self.dx * self.speed
        self.current_y += self.dy * self.speed

        for obj in interactions:
            # Note hit reward is a tuple, (reward for left, reward for right)
            hit, hit_reward, dx, dy = obj.check_intersect(self.current_x, self.current_y, self.dx, self.dy)
            if hit:
                self.dx = dx
                self.dy = dy
                self.step([])
                return hit_reward, obj
        return (0,0), None
    
    def render(self, viewer, scale):
        """ Render the ball (just a point) as a 10x10 pixel cube
        centered at location
        """
        l = (self.current_x - 5)*scale
        r = (self.current_x + 5)*scale
        t = (self.current_y - 5)*scale
        b = (self.current_y + 5)*scale
        v = [(l, b), (l, t), (r, t), (r, b)]
        if self.geom is None:
            self.geom = rendering.FilledPolygon(v)
            viewer.add_geom(self.geom)
        self.geom.v = v
        return self.geom

class PongBlock:
    def __init__(self, l=0, r=1, t=1, b=0, nx = 1, ny=-1, hit_reward=0):
        self.l = l
        self.r = r
        self.t = t
        self.b = b
        self.center_y = (t+b)/2
        self.center_x = (r+l)/2
        self.geom = None
        self.updated = False
        self.nx = nx
        self.ny = ny
        self.hit_reward = hit_reward
        self.last_hit = 0


    
    def step(self):
        self.last_hit += 1
    
    def render(self, viewer, scale):
        l = (self.l)*scale
        r = (self.r)*scale
        t = (self.t)*scale
        b = (self.b)*scale
        v = [(l, b), (l, t), (r, t), (r, b)]
        if self.geom is None:
            self.geom = rendering.FilledPolygon(v)
            viewer.add_geom(self.geom)
        if self.updated:
            self.geom.v = v
            self.updated = False
        return self.geom
    
    def check_intersect(self, ix, iy, current_dx=None, current_dy=None):
        hit = (self.l < ix < self.r) and (self.b < iy < self.t)
        reward = (0, 0)
        dx = dy = None
        if hit:
            self.last_hit = 0
            dx = current_dx * self.nx
            dy = current_dy * self.ny
        return  hit, reward, dx, dy

class PongPaddle:
    def __init__(self, l=0, r=1, t=1, b=0, speed=1.0, nx = 1, ny=-1, hit_reward_r=0, hit_reward_l=0, seed=0):
        self.l = l
        self.r = r
        self.t = t
        self.b = b
        self.center_y = (t+b)/2
        self.center_x = (r+l)/2
        self.speed = speed
        self.geom = None
        self.updated = False
        self.nx = nx
        self.ny = ny
        self.hit_reward_r = hit_reward_r
        self.hit_reward_l = hit_reward_l
        self.last_hit = 0
        self.rng = np.random.RandomState(seed)


    
    def step(self, bounds, dx, dy):
        self.t += dy
        self.b += dy
        if self.b < bounds[2]:
            self.t += abs(dy)
            self.b += abs(dy)
        elif self.t > bounds[3]:
            self.t -= abs(dy)
            self.b -= abs(dy)
        self.updated = True
        self.last_hit += 1
    
    def render(self, viewer, scale):
        l = (self.l)*scale
        r = (self.r)*scale
        t = (self.t - 5)*scale # Shorter so interaction with ball is not weird
        b = (self.b + 5)*scale # Shorter so interaction with ball is not weird
        v = [(l, b), (l, t), (r, t), (r, b)]
        if self.geom is None:
            self.geom = rendering.FilledPolygon(v)
            viewer.add_geom(self.geom)
        if self.updated:
            self.geom.v = v
            self.updated = False
        return self.geom
    
    def check_intersect(self, ix, iy, current_dx=None, current_dy=None):
        hit = (self.l < ix < self.r) and (self.b < iy < self.t)
        reward_l = self.hit_reward_l if self.last_hit > 100 else 0
        reward_r = self.hit_reward_r if self.last_hit > 100 else 0
        dx = dy = None

        # The hit direction is based on where the hit occured.
        # At the top, we reflect upwards and vice versa at the bottom.
        # At the center, we reflect straight back but this leads to the AI just bouncing back and forth!
        # So we make it so that there is a minimum angle, so cannot hit straight back.
        if hit:
            self.last_hit = 0
            theta = (self.t - iy) / (self.t - self.b) * math.pi
            dx = math.copysign(max(math.sin(theta), 0.25), self.nx * current_dx)
            dy = math.cos(theta)
            dy = math.copysign(max(abs(dy), 0.1), dy)
        return  hit, (reward_l, reward_r), dx, dy

        
class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()