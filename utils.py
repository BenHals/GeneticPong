import math
import gym
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy import invert
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

from customPong import CustomPong


def multi_agent_step(self, action):
    self._before_step(action)
    observation, (reward_l, reward_r), done, info = self.env.step(action)
    done = self._after_step(observation, reward_r, done, info)

    return observation, (reward_l, reward_r), done, info
def single_agent_step(self, action):
    self._before_step(action)
    observation, reward, done, info = self.env.step(action)
    done = self._after_step(observation, reward, done, info)

    return observation, reward, done, info

def show_video():
  """ Show recorded videos in the notebook
  """
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    for mp4 in sorted(mp4list, key = lambda fn: float(fn.split('video')[3].split('.mp4')[0])):
      video = io.open(mp4, 'r+b').read()
      encoded = base64.b64encode(video)
      ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                  loop controls style="height: 400px;">
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def record_env(env):
  """ Create a recorded environment.
  Videos are saved to the ./video folder
  """
  Monitor.step = multi_agent_step
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env

def record_env_walker(env):
  """ Create a recorded environment.
  Videos are saved to the ./video folder
  """
  Monitor.step = single_agent_step
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env


def recordPongGame(player_left, player_right, max_steps=5000):  
    print("\n Recording Game")
    print("---------------------")

    # Setup the game
    env = record_env(CustomPong())
    observation_left, observation_right = env.reset()

    # We record the score, or 'reward' for each player
    totalReward_left = 0
    totalReward_right = 0

    # The game is run in steps, where each player gets to pick their action
    for step in range(max_steps):
        # In this pong game, players only get to pick their action every 10 steps
        if step % 10 == 0:
            # The players decides their action based on their senses 'observing' the world
            action_left = player_left.getOutput(observation_left)
            action_right = player_right.getOutput(observation_right)

        # We progress the game one step based on the selected actions of each player.
        # The game sends back the next observation each player makes, as well as their rewards
        (observation_left, observation_right), (reward_left, reward_right), done, info = env.step((action_left, action_right))
        totalReward_left += reward_left
        totalReward_right += reward_right
        if done:
            break
    print(f"Game Finished! Left got a score of {totalReward_left}, Right got a score of {totalReward_right}")
    env.close()

def displayMultiAgentGame(left_agent, right_agent, env_type, max_steps=5000, monitor=False):  
    env = env_type()
    if monitor:
        print("\n Recording Game")
        print("---------------------")
        env = record_env(env)

    # Setup the game
    observation_left, observation_right = env.reset()

    # We record the score, or 'reward' for each player
    totalReward_left = 0
    totalReward_right = 0

    # The game is run in steps, where each player gets to pick their action
    for step in range(max_steps):
        # In this pong game, players only get to pick their action every 10 steps
        if step % 10 == 0:
            # The players decides their action based on their senses 'observing' the world
            action_left = left_agent.getOutput(observation_left)
            action_right = right_agent.getOutput(observation_right)
        if not monitor:
            env.render()
        # We progress the game one step based on the selected actions of each player.
        # The game sends back the next observation each player makes, as well as their rewards
        (observation_left, observation_right), (reward_left, reward_right), done, info = env.step((action_left, action_right))
        totalReward_left += reward_left
        totalReward_right += reward_right
        if done:
            break
    print(f"Game Finished! Left got a score of {totalReward_left}, Right got a score of {totalReward_right}")
    env.close()

def recordWalker(walker, max_steps=5000):  
    print("\n Recording Walker")
    print("---------------------")

    # Setup the game
    # We are going to use an environemnt from OpenAI
    # This is a single agent environment, so instead of observations, actions and rewards for each side, we only have one.
    # This is the only difference!

    env = record_env_walker(gym.make('BipedalWalker-v3'))
    observation = env.reset()

    # We record the score, or 'reward' for the walker
    totalReward = 0
    # The game is run in steps, where each player gets to pick their action
    for step in range(max_steps):
        # In the walker environment, the AI picks an action every step
        # The walker decides its action based on their senses 'observing' the world
        action = walker.getOutput(observation)

        # We progress the game one step based on the selected action of the walker.
        # The game sends back the next observation
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            break
    print(f"Game Finished! The walker got a score of {totalReward}")
    env.close()


def displaySingleAgentGame(nn, env_type, max_steps=10000, monitor=False):  
    env = env_type()
    if monitor:
        print("\n Recording Walker")
        print("---------------------")
        env = record_env_walker(env)

    ob = env.reset()
    totalReward = 0
    for step in range(max_steps):
        if not monitor:
            env.render()
        if step % 10 == 0:
            a = nn.getOutput(ob)
        ob, reward, done, info = env.step(a)
        totalReward += reward
        if done:
            break
    env.reset()
    print(f"Agent: Expected Fitness of {nn.fitness} | Actual Fitness = {totalReward}")
    env.close()

def versus_match(player_left, player_right, max_steps=5000):
    """ Play a multi-agent game, playing the left player against the right
    """
    # Setup the game
    # You might notice we don't record here. We will be playing many games
    # so recording would only slow us down!
    env = CustomPong()
    observation_left, observation_right = env.reset()

    # We record the score, or 'reward' for each player
    totalReward_left = 0
    totalReward_right = 0

    # The game is run in steps, where each player gets to pick their action
    for step in range(max_steps):
        # In this pong game, players only get to pick their action every 10 steps
        if step % 10 == 0:
            # The players decides their action based on their senses 'observing' the world
            action_left = player_left.getOutput(observation_left)
            action_right = player_right.getOutput(observation_right)

        # We progress the game one step based on the selected actions of each player.
        # The game sends back the next observation each player makes, as well as their rewards
        (observation_left, observation_right), (reward_left, reward_right), done, info = env.step((action_left, action_right))
        totalReward_left += reward_left
        totalReward_right += reward_right
        if done:
            break
    return totalReward_left, totalReward_right

def evaluate_walker(walker, max_steps=5000):  
    # Setup the game
    # We are going to use an environemnt from OpenAI
    # This is a single agent environment, so instead of observations, actions and rewards for each side, we only have one.
    # This is the only difference!

    env = gym.make('BipedalWalker-v3')
    observation = env.reset()

    # We record the score, or 'reward' for the walker
    totalReward = 0
    # The game is run in steps, where each player gets to pick their action
    for step in range(max_steps):
        # In the walker environment, the AI picks an action every step
        # The walker decides its action based on their senses 'observing' the world
        action = walker.getOutput(observation)

        # We progress the game one step based on the selected action of the walker.
        # The game sends back the next observation
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            break
    return totalReward





def draw_node(ax, bb, bt, bl, br, shape, text):
    width = br-bl
    height = bt-bb
    cy = (bb + bt) / 2
    cx = (bl + br) / 2
    margin = min(min(width, height)*0.1, 0.025)
    box_sides = min(width-margin, height-margin)
    radius = box_sides/2
    b = cy - box_sides/2
    t = cy + box_sides/2
    l = cx - box_sides/2
    r = cx + box_sides/2
    if shape in ['rect', 'r']:
        p = patches.Rectangle((0, b), box_sides, box_sides,
                                fill=False)
    else:
        p = patches.Circle((cx, cy), radius=radius,
                                fill=True, facecolor='white', edgecolor='black')

    ax.add_patch(p)
    ax.annotate(text, (cx, cy), ha='center', va='center')

    return cx, cy

def show_neuron(neuron, test_input):
    fig, ax = plt.subplots(figsize=(5, 5))

    nx, ny = draw_node(ax, 0, 1, 0.4, 0.6, 'c', f"b={neuron.bias}")

    input_box_height = 1 / len(test_input)
    input_positions = [(input_box_height*(i)) for i in range(len(test_input))]
    for i, p, w in zip(test_input, input_positions, neuron.connection_weights):
        cx, cy = draw_node(ax, p, p+input_box_height, 0, 0.2, 'r', i)
        dx = nx-cx
        dy = ny-cy
        sdx = (0.2/dx)
        dx = dx*sdx
        dy = dy*sdx
        theta = math.degrees(math.atan(dy/dx))
        sx = 0.2
        sy = cy
        ex = sx+dx
        ey = sy + dy
        arrow = patches.Arrow(sx, sy, dx, dy, width=0.05,
                                color='black')
        
        ax.annotate(w, (ex, ey+0.025), ha='right', va='bottom', rotation=theta, rotation_mode= "anchor")
        ax.add_patch(arrow)

    output_val = sum([w*i for i, w in zip(test_input, neuron.connection_weights)]) + neuron.bias
    sx = 0.6
    sy = ny
    dx = 0.2
    dy = 0
    ex = sx+dx
    ey = sy+dy
    arrow = patches.Arrow(sx, sy, dx, dy, width=0.05,
                            color='black')

    ax.annotate("Output", (ex, ey+0.025), ha='right', va='bottom')
    ax.annotate(f"{'+'.join([f'{w}x{i}' for i, w in zip(test_input, neuron.connection_weights)])} \n + {neuron.bias} = {output_val}", (1.0, ey-0.025), ha='right', va='top')
    ax.add_patch(arrow)
    ax.set_axis_off()
    plt.show()


def show_network(nn, test_input):
    fig, ax = plt.subplots(figsize=(5, 5))
    layer_width = 1 / (len(nn.nodeCount)*2 - 1)
    input_box_height = 1 / len(test_input)
    input_positions = [(input_box_height*(i)) for i in range(len(test_input))]
    layer_locations = []
    for i, p in zip(test_input, input_positions):
        cx, cy = draw_node(ax, p, p+input_box_height, 0, layer_width, 'r', i)
        layer_locations.append((cx, cy, i))

    for li, l in enumerate(nn.neuron_layers):
        node_height = 1 / len(l)
        input_positions = [(node_height*(i)) for i in range(len(l))]
        next_layer_locations = []
        layer_start_x = 2*(li+1)*layer_width
        layer_end_x = (2*(li+1)+1)*layer_width
        prev_layer_start_x = (2*(li+1)-1)*layer_width
        for n, p in zip(l, input_positions):
            nx, ny = draw_node(ax, p, p+node_height, layer_start_x, layer_end_x, 'c', "")
            v = 0
            for i, (cx, cy, val) in enumerate(layer_locations):
                w = n.connection_weights[i]
                v += val*w
                dx = nx-prev_layer_start_x
                dy = ny-cy
                # sdx = (layer_width/dx)
                # dx = dx*sdx
                # dy = dy*sdx
                theta = math.degrees(math.atan(dy/dx))
                sx = prev_layer_start_x
                sy = cy
                ex = sx+dx
                ey = sy + dy
                arrow = patches.Arrow(sx, sy, dx, dy, width=0.005,
                                        color='black', alpha=0.1)
                
                ax.annotate("", (ex, ey+0.025), ha='right', va='bottom', rotation=theta, rotation_mode= "anchor")
                ax.add_patch(arrow)
            v += n.bias
            next_layer_locations.append((nx, ny, v))
            nx, ny = draw_node(ax, p, p+node_height, layer_start_x, layer_end_x, 'c', f"{v:.0f}")
        layer_locations = next_layer_locations
    ax.invert_yaxis()
    ax.set_axis_off()
    plt.show()
