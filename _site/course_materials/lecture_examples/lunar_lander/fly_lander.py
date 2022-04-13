import random
import numpy as np
import os
import tensorflow as tf
import gym
import collections
import pandas as pd
import getpass
import time
import sys

current_user = getpass.getuser()
if(current_user == "gregryslik"):
    model_path = os.path.expanduser("~/gitRepos/osu_cse3521_spring2022/course_materials/lecture_examples/lunar_lander/models/")
    base_path = os.path.expanduser("~/gitRepos/osu_cse3521_spring2022/course_materials/lecture_examples/lunar_lander/")
else:
    sys.exit("Wrong os selected")

################################################
# Record good model
#################################################
model_file_path = model_path + "episode-480_model_success.h5"
model = tf.keras.models.load_model(model_file_path)

env = gym.make('LunarLander-v2')
#env = gym.wrappers.Monitor(env, base_path+"/video/good_model_492/")

state = env.reset().reshape(1, 8)
episode_reward = 0
step = 0
done = False
while not done:  # will auto terminate when it reaches 200
    prediction_values = np.array(model.predict_on_batch(state))
    action = np.argmax(prediction_values)
    state, reward, done, info = env.step(action)
    state = state.reshape(1, 8)
    step += 1
    episode_reward += reward
    env.render()
env.close()

print("Episode Reward: " + str(episode_reward))