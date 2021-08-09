from gym.wrappers import Monitor

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