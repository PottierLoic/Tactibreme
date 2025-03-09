import torch
from torch import nn, Tensor

class Network(nn.Module):
    def __init__(self, input_channels: int = 8, board_size: tuple[int, int] = (5, 5)) -> None:
        super(Network, self).__init__()
        self.input_channels = input_channels
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # NOTE: 3x3 kernel will maybe not work well for a 5x5 board, need to test

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(board_size[0] * board_size[1] * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        features = self.conv_layers(x)
        output = self.fc_layers(features)
        return output

