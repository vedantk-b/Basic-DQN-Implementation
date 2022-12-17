import torch
import math
import gymnasium as gym
import nns
from utils import ReplayBuffer
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("LunarLander-v2")
observation, info = env.reset()
print(f"initially {observation, info}")

Q_NET = nns.NN(8, 4)
TARG_Q_NET = nns.NN(8, 4)
TARG_Q_NET.load_state_dict(Q_NET.state_dict())

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

EPISODES = 1000
MAX_STEPS = 100
EPS_ST = 0.95
EPS_EN = 0.05
BATCH_SIZE = 128
EPS_DECAY = 10000
DISCOUNT_FACTOR = 0.99
D = ReplayBuffer(1000, batch_size=BATCH_SIZE)
LR = 3e-4

optimizer = torch.optim.Adam(Q_NET.parameters(), lr=LR)
criterion = torch.nn.HuberLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"

rewlis = []

dec_step = 0
for episode in range(EPISODES):
    terminated = False
    steps = 0
    eps = EPS_EN + (EPS_ST - EPS_EN)*math.exp(-1*dec_step/EPS_DECAY)
    eprew = 0
    while not terminated and steps < MAX_STEPS:
        observation, info = env.reset()
        steps += 1
        dec_step += 1
        sample = torch.rand(1)
        if(sample < eps):
            action = env.action_space.sample()
        else:
            action = Q_NET.forward(torch.tensor(observation)).argmax().item()
        state = observation
        observation, reward, terminated, truncated, info = env.step(action)
        eprew += reward
        D.store(Transition(state, action, observation, reward))

        if len(D) > BATCH_SIZE and dec_step > 500:
            batch = D.fetch()
            batch = Transition(*zip(*batch))
            state_batch = torch.tensor(np.array(batch.state))
            action_batch = torch.tensor(np.array(batch.action))
            reward_batch = torch.tensor(np.array(batch.reward))
            next_state_batch = torch.tensor(np.array(batch.next_state))

            if terminated or truncated:
                target = reward_batch
            else:
                target = reward_batch + DISCOUNT_FACTOR*TARG_Q_NET(next_state_batch).max(1).values

            mask_target = Q_NET(state_batch)

            for i in range(BATCH_SIZE):
                mask_target[i][action_batch[i]] = target[i]

            loss = criterion(Q_NET(state_batch), mask_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if((dec_step+1)%100 == 0):
                TARG_Q_NET.load_state_dict(Q_NET.state_dict())
    eprew/=steps
    print("episode reward", eprew)
    rewlis.append(eprew)


    if((EPISODES+1)%100 == 0):
        torch.save(Q_NET.state_dict())

plt.plot(eprew)
plt.show()
plt.savefig()