import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl

class CustomPong(gym.Env):
    """
    Description:
        Pong Game
    Observation:
        Type: Box(3)
        Num     Observation                                     Min                     Max
        0       Ball horizintal distance from paddle            0                       w (512)
        1       Paddle Height                                   0                       h (256)
        2       Ball height                                     0                       h (256)
    Actions:
        Type: Discrete(3)
        Num   Action
        0     Move paddle down
        1     Move paddle up
        2     Do nothing
    Reward:
        Reward is 1 for hitting the ball the the paddle.
        Missing the ball incurs a loss proportional to the miss distance
    Starting State:
        Ball is initialized in the center, with max x and y velocity
    Episode Termination:
        3 Failed balls
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.ball_speed = 3.0
        self.paddle_speed = 3.0
        self.paddle_length = 50
        self.paddle_hit_reward = 0.1
        self.w = 512
        self.h = 256
        self.bounds = [0, self.w, 0, self.h]
        self.center_bounds = [30, self.w-30, 30, self.h-30]
        
        high = np.array(
            [
                self.w, # Ball distance
                self.h, # Paddle height
                self.h, # Ball height
                1, # Ball dx
                1, # Ball dy
                # math.pi, # Ball Angle (radians)
            ],
            dtype=np.float32,
        )
        self.state_r = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.state_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.reset()

        self.action_space = spaces.Discrete(3) # Up, Down, Nothing
        self.observation_space = spaces.Box(np.zeros(high.shape[0]), high, dtype=np.float32)

        self.viewer = None
        

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_l, action_r):
        a_l = np.argmax(action_l)
        a_r = np.argmax(action_r)

        self.last_failed_l += 1
        self.last_failed_r += 1
        
        in_center = (self.center_bounds[0] < self.ball.current_x < self.center_bounds[1]) and (self.center_bounds[2] < self.ball.current_y < self.center_bounds[3])

        (self.reward_l, self.reward_r), hit_obj = self.ball.step(self.interactions if not in_center else [])
        if hit_obj == self.paddle_l:
            self.paddle_last_hit = self.paddle_l
        elif hit_obj == self.paddle_r:
            self.paddle_last_hit = self.paddle_r

        
        dy = self.paddle_speed *(1 - a_r)
        self.paddle_r.step(self.bounds, dx=0, dy=dy)

        dy = self.paddle_speed *(1 - a_l)
        self.paddle_l.step(self.bounds, dx=0, dy=dy)
        # if action == 0:
        #     self.paddle_r.step([0, self.w, self.h, 0], dx=0, dy=self.paddle_speed)
        # if action == 1:
        #     self.paddle_r.step([0, self.w, self.h, 0], dx=0, dy=-self.paddle_speed)
        # e_AI_dy = math.copysign(self.paddle_speed, self.ball.current_y - self.paddle_l.center_y)
        # e_AI_dy = self.paddle_speed if self.ball.current_y > self.paddle_l.center_y else -1 * self.paddle_speed if self.ball.current_y < self.paddle_l.center_y else 0
        # self.paddle_l.step(bounds, dx=0, dy=e_AI_dy)

        if self.ball.current_x >= self.w-6:
            # self.done = True
            hp_loss = (((min(abs(self.paddle_r.t - self.ball.current_y), abs(self.paddle_r.b - self.ball.current_y)) / (self.h - (self.paddle_r.t - self.paddle_r.b)))))
            self.reward_r -=  hp_loss
            if self.paddle_last_hit == self.paddle_l:
                self.reward_l += hp_loss
                # self.wall_hits_r += 2
            # self.wall_hits_r += 1
            self.hp_r -= hp_loss * 100
            self.ball.dx = -1 * self.ball.dx
            self.ball.current_x = self.w - 11
            self.paddle_last_hit = None
            self.last_failed_r = 0
            # input(f"{self.ball.current_x}-{self.ball.current_y} : {self.paddle_r.t}-{self.paddle_r.b}")
        if self.ball.current_x <= 4:
            # self.done = True
            hp_loss = (((min(abs(self.paddle_l.t - self.ball.current_y), abs(self.paddle_l.b - self.ball.current_y)) / (self.h - (self.paddle_l.t - self.paddle_l.b)))))
            self.reward_l -=  hp_loss
            if self.paddle_last_hit == self.paddle_r:
                self.reward_r += hp_loss
                # self.wall_hits_l += 2
            # self.wall_hits_l += 1
            self.hp_l -= hp_loss * 100
            # self.reward -= 1 - min(abs(self.paddle_r.t - self.ball.current_y), abs(self.paddle_r.b - self.ball.current_y)) / (self.h - (self.paddle_r.t - self.paddle_r.b))
            self.ball.dx = -1 * self.ball.dx
            self.ball.current_x = 11
            self.paddle_last_hit = None
            self.last_failed_l = 0
            # self.reward += 1
            # self.wall_hits += 1
        
        # if max(self.wall_hits_l, self.wall_hits_r) > 2:
        if min(self.hp_l, self.hp_r) <= 0:
            self.done = True

        self.set_state()

        self.reward_l -= 0.0001
        self.reward_r -= 0.0001

        self.total_reward_l += self.reward_l
        self.total_reward_r += self.reward_r
        return (self.state_l, self.state_r), (self.reward_l, self.reward_r), self.done, None

    def set_state(self):
        self.state_r[0] = (self.w - self.ball.current_x) / self.w    
        self.state_r[1] = ((self.paddle_r.t + self.paddle_r.b) / 2) / self.h
        self.state_r[2] = (self.ball.current_y) / self.h
        self.state_r[3] = (self.ball.dx + 1) / 2
        self.state_r[4] = (self.ball.dy + 1) / 2

        self.state_l[0] = (self.ball.current_x) / self.w    
        self.state_l[1] = ((self.paddle_l.t + self.paddle_l.b) / 2) / self.h
        self.state_l[2] = (self.ball.current_y) / self.h
        self.state_l[3] = (self.ball.dx * -1 + 1) / 2
        self.state_l[4] = (self.ball.dy + 1) / 2

    def reset(self):
        seed = np.random.randint(10000)
        self.rng = np.random.RandomState(seed)
        top_wall = PongBlock(l=0, r=self.w, t=self.h, b=self.h-10)
        bottom_wall = PongBlock(l=0, r=self.w, t=10, b=0)
        self.paddle_l = PongPaddle(l=0, r=10, t = (self.h//2) + (self.paddle_length//2), b= (self.h//2) - (self.paddle_length//2), ny=1, nx=-1, hit_reward_l=self.paddle_hit_reward, speed=self.paddle_speed, seed=seed)
        self.paddle_r = PongPaddle(l=self.w-10, r=self.w, t = (self.h//2) + (self.paddle_length//2), b= (self.h//2) - (self.paddle_length//2), ny=1, nx=-1, hit_reward_r=self.paddle_hit_reward, speed=self.paddle_speed, seed=seed)
        self.ball = PongBall(x=self.w//10, y = self.h//2, speed=self.ball_speed, dx=self.rng.rand()*0.75 + 0.25, dy=self.rng.rand()*0.75 + 0.25, seed=seed)
        self.interactions = [top_wall, bottom_wall, self.paddle_l, self.paddle_r]
        self.steps_beyond_done = None
        self.wall_hits_l = 0
        self.wall_hits_r = 0
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
        
        # self.state = np.array([self.h - self.ball.current_x, (self.paddle_r.t + self.paddle_r.b) //2, self.ball.current_y])
        self.set_state()
        return self.state_l, self.state_r

    def render(self, mode="human"):
        screen_width = self.w
        screen_height = self.h

        world_width = self.w
        scale = screen_width / world_width

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.score_label_l = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=0+11,
                y=self.h - 20,
                anchor_x="left",
                anchor_y="top",
                color=(1, 1, 255, 255),
            ))
            self.score_label_r = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=self.w-11,
                y=self.h - 20,
                anchor_x="right",
                anchor_y="top",
                color=(1, 1, 255, 255),
            ))
            self.hp_label_l = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=0+11,
                y=self.h - 40,
                anchor_x="left",
                anchor_y="top",
                color=(255, 1, 50, 255),
            ))
            self.hp_label_r = DrawText(pyglet.text.Label(
                "0000",
                font_size=16,
                x=self.w-11,
                y=self.h - 40,
                anchor_x="right",
                anchor_y="top",
                color=(255, 1, 50, 255),
            ))
            self.viewer.add_geom(self.score_label_l)
            self.viewer.add_geom(self.score_label_r)
            self.viewer.add_geom(self.hp_label_l)
            self.viewer.add_geom(self.hp_label_r)

        for obj in self.interactions:
            rend = obj.render(self.viewer)
        
        if self.last_failed_l < 10:
            self.paddle_l.geom.set_color(0.9, 0.0, 0.0)
        else:
            self.paddle_l.geom.set_color(0.0, 0.0, 0.0)
        if self.last_failed_r < 10:
            self.paddle_r.geom.set_color(0.9, 0.0, 0.0)
        else:
            self.paddle_r.geom.set_color(0.0, 0.0, 0.0)

        rend = self.ball.render(self.viewer)
        self.score_label_l.label.text = f"{self.total_reward_l:.2f}"
        self.score_label_r.label.text = f"{self.total_reward_r:.2f}"
        self.hp_label_l.label.text = f"{self.hp_l:.0f}"
        self.hp_label_r.label.text = f"{self.hp_r:.0f}"

        r = self.viewer.render(return_rgb_array=mode == "rgb_array")
        
        # self.score_label.draw()

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
        norm = np.linalg.norm([dx, dy])
        self.dx = dx / norm
        self.dy = dy / norm
        self.geom = None
        self.trans = None
        self.rng = np.random.RandomState(seed)

    
    def step(self, interactions):
        self.current_x += self.dx * self.speed
        self.current_y += self.dy * self.speed
        # self.current_x += self.dx * (self.rng.rand() * 0.5)
        # self.current_y += self.dy * (self.rng.rand() * 0.5)

        for obj in interactions:
            hit, hit_reward, dx, dy = obj.check_intersect(self.current_x, self.current_y, self.dx, self.dy)
            if hit:
                self.dx = dx
                self.dy = dy
                self.step([])
                return hit_reward, obj
        return (0,0), None
    
    def render(self, viewer):
        l = self.current_x - 5
        r = self.current_x + 5
        t = self.current_y - 5
        b = self.current_y + 5
        # l = -5
        # r = 5
        # t = -5
        # b = 5
        v = [(l, b), (l, t), (r, t), (r, b)]
        if self.geom is None:
            self.geom = rendering.FilledPolygon(v)
            # self.trans = rendering.Transform()
            # self.geom.add_attr(self.trans)
            viewer.add_geom(self.geom)
        self.geom.v = v
        # self.trans.set_translation(self.current_x, self.current_y)
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
    
    def render(self, viewer):
        l = self.l
        r = self.r
        t = self.t
        b = self.b
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
        # reward = self.hit_reward if self.last_hit > 100 else 0
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
        # self.l += dx
        # self.r += dx
        self.t += dy
        self.b += dy
        # if self.l < bounds[0]:
        #     self.l += abs(dx)
        #     self.r += abs(dx)
        # if self.r > bounds[1]:
        #     self.l -= abs(dx)
        #     self.r -= abs(dx)
        if self.b < bounds[2]:
            self.t += abs(dy)
            self.b += abs(dy)
        elif self.t > bounds[3]:
            self.t -= abs(dy)
            self.b -= abs(dy)
        # self.center_y = (self.t+self.b)/2
        # self.center_x = (self.r+self.l)/2
        self.updated = True
        self.last_hit += 1
    
    def render(self, viewer):
        l = self.l
        r = self.r
        t = self.t - 5 # Shorter so interaction with ball is not weird
        b = self.b + 5 # Shorter so interaction with ball is not weird
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
        if hit:
            self.last_hit = 0
            theta = (self.t - iy) / (self.t - self.b) * math.pi
            dx = math.copysign(max(math.sin(theta), 0.25), self.nx * current_dx)
            # dy = math.cos(theta + self.rng.rand()*0.01)
            dy = math.cos(theta)
            dy = math.copysign(max(abs(dy), 0.1), dy)
        return  hit, (reward_l, reward_r), dx, dy

        
class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()