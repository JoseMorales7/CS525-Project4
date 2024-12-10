import gym
from agents import experienceReplayBuffer_rainbow, RainbowAgent, QNetwork_rainbow
import torch
from agents import evaluate
from copy import deepcopy



if __name__ == "__main__":
    n_iter = 33000
    env = gym.make('gym_pvz:pvz-env-v2')
    nn_name = input("Save name: ")
    buffer = experienceReplayBuffer_rainbow(memory_size=100000, burn_in=10000)
    net = QNetwork_rainbow(env, device='cpu'
        , use_zombienet=False, use_gridnet=False)
    agent = RainbowAgent(env, net, buffer, n_iter=n_iter, batch_size=200)
    agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
    torch.save(agent.network, nn_name)
    agent._save_training_data(nn_name)