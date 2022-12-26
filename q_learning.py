import random

import gym
import matplotlib.pyplot as plt
import numpy as np


# if mode = True: learning rate vs average total reward, mode = False : discount rate vs average total reward
def train_agent(env, num_episodes, learning_rate, discount_rate, max_steps):
    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    max_epsilon = 1.0
    min_epsilon = 0.01
    epsilon = max_epsilon
    decay_rate = 0.005

    avg_reward_list = []

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False
        reward_list = []

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # take action and observe reward
            new_state, reward, done, info = env.step(action)
            reward_list.append(reward)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done:
                break

        avg_reward_list.append(np.mean(reward_list))
        # Decrease epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    print(f"Training completed for learning rate = {learning_rate} and discount rate = {discount_rate} over {num_episodes} episodes")
    return avg_reward_list


def plot_graphs(env, num_episodes, max_steps, mode):
    if mode:
        print("Learning Rate vs Convergence of Q-Learning")
    else:
        print("Discount Rate vs Convergence of Q-Learning")

    x_axis = [i for i in range(num_episodes)]

    variable_rate = 0
    rate_list = [0.1, 0.2, 0.4, 0.7, 1.0]
    for i in range(10):
        variable_rate += 0.1
        variable_rate = round(variable_rate, 2)
        for j in range(len(rate_list)):
            if mode:
                print(f"Training agent for the learning rate = {variable_rate} and discount rate = {rate_list[j]} over {num_episodes} episodes")
                plt.plot(x_axis, train_agent(env, num_episodes, variable_rate, rate_list[j], max_steps), label=f"Discount Rate = {rate_list[j]}")
            else:
                print(f"Training agent for the learning rate = {rate_list[j]} and discount rate = {variable_rate} over {num_episodes} episodes")
                plt.plot(x_axis, train_agent(env, num_episodes, rate_list[j], variable_rate, max_steps), label=f"Learning Rate = {rate_list[j]}")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Average Reward")
        if mode:
            plt.title(f"Convergence of Q-Learning for the Learning Rate = {variable_rate}")
        else:
            plt.title(f"Convergence of Q-Learning for the Discount Rate = {variable_rate}")
        plt.legend(loc="lower right")
        plt.show()


def main():
    # create Taxi environment
    env = gym.make('Taxi-v3')

    # training variables
    num_episodes = 5000
    max_steps = 200

    plot_graphs(env, num_episodes, max_steps, True)
    plot_graphs(env, num_episodes, max_steps, False)
    input("Press enter to exit")


if __name__ == "__main__":
    main()
