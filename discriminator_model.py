import torch.nn as nn

FILTER_SIZE = 64


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv_pipeline = self.create_conv_pipeline(input_shape)

    @staticmethod
    def create_conv_pipeline(input_shape):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=FILTER_SIZE, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTER_SIZE, out_channels=FILTER_SIZE * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER_SIZE * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTER_SIZE * 2, out_channels=FILTER_SIZE * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER_SIZE * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTER_SIZE * 4, out_channels=FILTER_SIZE * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER_SIZE * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTER_SIZE * 8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        conv_out = self.conv_pipeline(x)
        return conv_out.view(-1, 1).squeeze(dim=1)
