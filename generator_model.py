import torch.nn as nn

LATENT_VECTOR_SIZE = 100
FILTER_SIZE = 64


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        self.conv_pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=FILTER_SIZE * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(FILTER_SIZE * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTER_SIZE * 8, out_channels=FILTER_SIZE * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER_SIZE * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTER_SIZE * 4, out_channels=FILTER_SIZE * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER_SIZE * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTER_SIZE * 2, out_channels=FILTER_SIZE,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER_SIZE),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTER_SIZE, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv_pipe(x)
