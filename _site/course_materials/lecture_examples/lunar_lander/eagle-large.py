import random
import numpy as np
import os
import tensorflow as tf
import gym
import collections
import getpass
import time
import pandas as pd


current_user = getpass.getuser()
if(current_user == "gregryslik"):
    model_path = os.path.expanduser("~/gitRepos/osu_cse3521_spring2022/course_materials/lecture_examples/lunar_lander/models/")
else:
    model_path = '/home/ubuntu/data/code/lunar_lander/models/'


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = collections.deque(maxlen=1000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.model = self.create_model() # Will do the actual predictions

    def create_model(self):
        model = tf.keras.Sequential()
        state_shape = self.env.observation_space.shape #this returns (2,) as we observe a vector of 2 numbers (position and velocity)
        model.add(tf.keras.layers.Dense(150, input_dim = state_shape[0], activation = "relu"))
        model.add(tf.keras.layers.Dense(50, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)

        all_states = np.reshape([np.squeeze(x[0]) for x in samples], (batch_size, 8))
        all_actions = np.reshape([x[1] for x in samples], (batch_size, ))
        all_rewards = np.reshape([x[2] for x in samples], (batch_size, ))
        all_new_states = np.reshape([np.squeeze(x[3]) for x in samples], (batch_size, 8))
        all_dones = np.reshape([x[4] for x in samples], (batch_size, ))

        # Thus our target value is:
        future_discounted_rewards = np.array(self.model.predict_on_batch(all_new_states))  # A guess at the future discounted reward (the Q future table)
        future_max_reward = np.amax(future_discounted_rewards, axis=1)  # Figure out which reward we want -- pick the bigger one
        updated_future_discounted_rewards = all_rewards + self.gamma * future_max_reward * (~all_dones)

        all_targets = np.array(self.model.predict_on_batch(all_states))  # get us the predicted rewards
        all_targets[np.arange(len(all_targets)), np.array(all_actions)] = updated_future_discounted_rewards  # updated predicted rewards with future rewards

        # all states will predict the Q-value from our network. All targets is the q_value after taking the actiom
        # We want to make the Q value from our network prediction match our target
        self.model.train_on_batch(all_states, all_targets)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def save_model(self, fn):
        self.model.save(fn)



#Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#

#### State informaton
# X- position
# Y- position
# X-velocity
# Y-velocity
#lander angle
#angular velocity
#left leg ground
#right light ground

env = gym.make('LunarLander-v2')
num_episodes = 500
my_agent = DQN(env=env)
totalreward = []
steps = []
for episode in range(num_episodes):
    print("======================================================")
    print("Processing episode: " + str(episode))
    print("======================================================")
    time_start = time.time()
    cur_state = env.reset().reshape(1,8)
    episode_reward = 0
    step = 0
    while True: #will auto terminate when it reaches 200
        action = my_agent.act(cur_state)
        new_state, reward, done, info = env.step(action)
        new_state = new_state.reshape(1, 8)
        my_agent.remember(cur_state, action, reward, new_state, done)
        my_agent.replay()
        cur_state = new_state
        episode_reward += reward
        step +=1
        if done:
            break
    totalreward.append(episode_reward)
    steps.append(step)
    print("--------------------------------------------------------")
    print("Episode: " + str(int(episode)) + " completed in: " + str(step) + " steps.")
    print("--------------------------------------------------------")
    if episode_reward < 200.0:
        print("Failed to complete episode: " + str(episode) + " with a total reward of: " + str(episode_reward))
        if episode % 10 == 0:
            my_agent.save_model(model_path + "episode-{}_model_failure.h5".format(episode))
    else:
        print("Successfully completed in episode: " + str(episode) + " with a total reward of: " + str(episode_reward))
        my_agent.save_model(model_path + "episode-{}_model_success.h5".format(episode))
    time_end = time.time()
    tf.keras.backend.clear_session()
    print("Processing episode: " + str(episode) + " took: " + str(int(time_end - time_start)) + " seconds. Avg running reward is: " + str(np.array(totalreward)[-100:].mean()))
env.close()

results_df = pd.DataFrame(totalreward, columns = ['episode_reward'])
results_df['steps_taken'] = steps
results_df['Success'] = results_df['episode_reward'] > 200.0
results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()

results_df.to_csv(model_path+"training_results.csv")
