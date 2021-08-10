import math
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy import invert


def wrap_env(env):
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env


def displayMultiAgentGame(nnl, nnr, env_type, max_steps=10000, monitor=False):  
    env = env_type()
    if monitor:
        env = wrap_env(env)
    ob_l, ob_r = env.reset()
    totalReward_l = 0
    totalReward_r = 0
    for step in range(max_steps):
        env.render()
        if step % 10 == 0:
            ar = nnr.getOutput(ob_r)
            al = nnl.getOutput(ob_l)
        (ob_l, ob_r), (r_l, r_r), done, info = env.step((al, ar))
        totalReward_l += r_l
        totalReward_r += r_r
        if done:
            break
    print(f"Left: Expected Fitness of {nnl.fitness} | Actual Fitness = {totalReward_l} ||| Right: Expected Fitness of {nnr.fitness} | Actual Fitness = {totalReward_r}")
    env.close()

def displaySingleAgentGame(nn, env_type, max_steps=10000, monitor=False):  
    env = env_type()
    if monitor:
        env = wrap_env(env)
    ob = env.reset()
    totalReward = 0
    for step in range(max_steps):
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
