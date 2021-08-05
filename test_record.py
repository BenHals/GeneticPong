import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
# import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

import cv2

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    for mp4 in sorted(mp4list, key = lambda fn: float(fn.split('video')[3].split('.mp4')[0])):
    #   video = io.open(mp4, 'r+b').read()
    #   encoded = base64.b64encode(video)
    #   ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
    #               loop controls style="height: 400px;">
    #               <source src="data:video/mp4;base64,{0}" type="video/mp4" />
    #           </video>'''.format(encoded.decode('ascii'))))
        cap = cv2.VideoCapture(mp4)
        ret, frame = cap.read()
        while(1):
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
                cap.release()
                cv2.destroyAllWindows()
                break
        cv2.imshow('frame',frame)
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env

def wrap_env(env):
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env


env = wrap_env(gym.make('BipedalWalker-v3'))
for i in range(3):
  observation = env.reset()

  count = 0
  while True:
    
      env.render()
      
      #your agent goes here
      action = env.action_space.sample() 
          
      observation, reward, done, info = env.step(action) 
    
          
      if done: 
        break;
      count += 1
            
env.close()
show_video()