import math
from torch import nn as nn


# Input shape: (batch_size, in_channels, height, width) = (N, 3, 224,224)
class SkinCancerModel(nn.Module):
    def __init__(self, in_channels=3, conv_channels=4):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm2d(3)

        self.block1 = SkinCancerBlock(in_channels, conv_channels)
        self.block2 = SkinCancerBlock(conv_channels, conv_channels * 2)
        self.block3 = SkinCancerBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = SkinCancerBlock(conv_channels * 4, conv_channels * 8)

        # Define multiple linear layers instead of a single one
        self.head = nn.Sequential(
            nn.Linear(6272, 1024),  # #224-> 112 -> 56 -> 28-> 14 :  (4*8) * 14 * 14 = 6,272 
            nn.ReLU(),
            nn.Linear(1024, 256),   
            nn.ReLU(),
            nn.Linear(256, 2)     
        )
        
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(block_out.size(0), -1)  # Flatten

        linear_output = self.head(conv_flat)

        return linear_output, self.head_softmax(linear_output)



class SkinCancerBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
