from collections import deque
import random

class ReplayBuffer:
    
    def __init__(self, capacity, batch_size) -> None:
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
    
    def store(self, x):
        self.memory.append(x)

    def fetch(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory) 