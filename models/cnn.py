import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, hidden_channel_num, output_number):
        super(CNN, self).__init__()
        self.hidden_channel_num = hidden_channel_num
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_channel_num, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel_num, out_channels=int(hidden_channel_num/2), kernel_size=2, stride=1)
        self.fc1 = nn.Linear(int(hidden_channel_num/2)*126, output_number)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, int(self.hidden_channel_num/2)*126)
        x = self.fc1(x)
        x = self.Sigmoid(x)
        return x