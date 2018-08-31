import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualModuleDouble(nn.Module):

    def __init__(self, inputSize, outputSize):

        super(ResidualModuleDouble, self).__init__()

        self.conv1 = nn.Conv2d(inputSize, inputSize, 3, padding=1)
        self.conv2 = nn.Conv2d(inputSize, outputSize, 3, padding=1)

    def forward(self, inputData):

        residual = torch.cat((inputData, inputData), 1)
        h = F.elu(self.conv1(inputData))
        h = self.conv2(h)
        h = residual + h
        o = F.elu(h)

        return o

class ResidualModuleSingle(nn.Module):

    def __init__(self, inputSize):

        super(ResidualModuleSingle, self).__init__()

        self.conv1 = nn.Conv2d(inputSize, inputSize, 3, padding=1)
        self.conv2 = nn.Conv2d(inputSize, inputSize, 3, padding=1)

    def forward(self, inputData):

        h = F.elu(self.conv1(inputData))
        h = self.conv2(h)
        h = inputData + h
        o = F.elu(h)

        return o

class VisionEncoder(nn.Module):

    def __init__(self):

        super(VisionEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 7, padding=3, stride=2)
        self.max1 = nn.MaxPool2d(2, 2)
        self.res1 = ResidualModuleDouble(32, 64)
        self.max2 = nn.MaxPool2d(2, 2)
        self.res2 = ResidualModuleDouble(64, 128)
        self.max3 = nn.MaxPool2d(2, 2)
        self.res3 = ResidualModuleDouble(128, 256)
        self.max4 = nn.MaxPool2d(2, 2)

    def forward(self, boards):

        h = F.leaky_relu(self.conv1(boards), 1e-2)
        h = self.max1(h)
        h = self.res1(h)
        h = self.max2(h)
        h = self.res2(h)
        h = self.max3(h)
        h = self.res3(h)
        o = self.max4(h)

        return o

class VisionDecoder(nn.Module):

    def __init__(self):

        super(VisionDecoder, self).__init__()

        self.upConv1 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.upConv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.upConv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.upConv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.upConv5 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1)

    def forward(self, featureVector):

        inputSize = torch.LongTensor([featureVector.size(2), featureVector.size(3)])
        h = F.elu(self.upConv1(featureVector, output_size=inputSize*2))
        h = F.elu(self.upConv2(h, output_size=inputSize*4))
        h = F.elu(self.upConv3(h, output_size=inputSize*8))
        h = F.elu(self.upConv4(h, output_size=inputSize*16))
        o = F.elu(self.upConv5(h, output_size=inputSize*32))

        return o

class PositionEstimator(nn.Module):

    def __init__(self):

        super(PositionEstimator, self).__init__()

        self.flatShape = 7 * 7 * 256
        self.lin1 = nn.Linear(self.flatShape, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 5)

    def forward(self, featureVector):

        h = featureVector.view(-1, self.flatShape)
        h = F.elu(self.lin1(h))
        h = F.elu(self.lin2(h))
        o = self.lin3(h)

        return o