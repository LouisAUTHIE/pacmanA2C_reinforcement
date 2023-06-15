import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import numpy as np
import gym

class A2CAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode='human')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.value_size = 1
        self.discount_factor = 0.99
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.005

        Actor = Sequential([
            Flatten(input_shape=self.state_size),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(9, activation="softmax", kernel_initializer='he_uniform')
        ])
        Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
        self.actor = Actor

        Critic = Sequential([
            Flatten(input_shape=self.state_size),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, kernel_initializer='he_uniform')
        ])
        Critic.compile(loss='mse', optimizer=RMSprop(lr=0.005))
        self.critic = Critic

    def get_action(self, state):
        policy = self.actor.predict(np.array([state]), verbose=0)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(np.array([state]),verbose=0)[0]
        next_value = self.critic.predict(np.array([next_state]), verbose=0)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(np.array([state]), np.array(advantages), epochs=1, verbose=0)
        self.critic.fit(np.array([state]), np.array(target), epochs=1, verbose=0)

    def run(self):
        scores, episodes = [], []
        for episode in range(7):
            done = False
            score = 0
            state = self.env.reset()

            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.train_model(state, action, reward, next_state, done)
                score += reward
                state = next_state
                print("Score:", score, "Episode:", episode, "Action:", action, "Reward:", reward, "Done:", done, "Info:", info)

            scores.append(score)
            episodes.append(episode)

            print("Episode:", episode, " Score:", score)
        print(scores)
        self.env.close()

agent = A2CAgent("MsPacman-v0")
agent.run()