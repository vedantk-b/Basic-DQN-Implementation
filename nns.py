import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):

    def __init__(self, observation_shape, action_shape) -> None:
        super().__init__()
        self.layer1 = nn.Linear(observation_shape, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_shape)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = (self.layer3(x))
        return x

if __name__ == "__main__":
    q_net = NN(8, 4)
    X = torch.rand(9, 8)
    out = q_net.forward(X)
    print(out)
    print(out.argmax(1))