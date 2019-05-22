import torch
import torch.nn as nn
import numpy as np
# import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls


class ResNetConv(ResNet):    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        intermediate = []
        x = self.layer1(x); intermediate.append(x)
        x = self.layer2(x); intermediate.append(x)
        x = self.layer3(x); intermediate.append(x)
        
        return x, intermediate


# ResNet model to perform Human Pose Estimation using Regression method
# Returns outputs of the form [N*34] i.e., x,y coordinates of 17 keypoints
class ResNetModelR(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])

        # other network modules
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 34)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, inputs):
        x, _ = self.res_conv(inputs)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + 0.5
        return x


class Softargmax(nn.Module):
    """
    Module for Softargmax
    Returns the coordinates for C channels in form [x1, y1, x2, y2, ... xC, yC]
    Input shape: N*C*H*W
    Output shape: N*2C
    TODO: Make this more generalized
    """
    def __init__(self, dim):
        super(Softargmax, self).__init__()
        self.sftmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x = self.sftmax(x.view(x.shape[:-2] + (-1,))).view(x.shape)
        # calculating weights
        w = torch.arange(x.shape[-1]).repeat(np.product(x.shape[:-1]), 1).view(x.shape).float() / x.shape[-1]
        w = w.to(x.device)
        # calculating keypoints in 'y'
        dy = torch.sum(x * w, dim=(2, 3))
        # calculating keypoints in 'x'
        w = w.transpose(2, 3)
        dx = torch.sum(x * w, dim=(2, 3))
        # reshaping dx and dy into the expected vector format i.e. B*2K
        x = torch.cat((dx, dy), dim=0).transpose(0, 1).contiguous().view(-1, dx.shape[0]).transpose(0, 1)
        return x


# ResNet model to perform Human Pose Estimation using Softargmax method
# Returns outputs of the form [N*34] i.e., x,y coordinates of 17 keypoints
class ResNetModelS(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])

        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        # remove the final 2 layers from resnet for deconvolution
        self.res_conv = nn.Sequential(*(list(self.res_conv.children())[:-2]))

        # defining decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 17, kernel_size=4, stride=4),
            nn.BatchNorm2d(17)
            # nn.MaxPool2d(kernel_size=2)
        )

        self.softargmax = Softargmax(dim=2)

    def forward(self, inputs):
        x = self.res_conv(inputs)
        x = self.decoder(x)
        x = self.softargmax(x)
        return x


class Upsample(nn.Module):
    """
    Class to upsample a given matrix
    NOTE: nn.Upsample depricated - adding wrapper around functional interpolate
    """
    def __init__(self, size, mode):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


# ResNet model to perform Semantic Segmentation using only a single upsampling layer
# Returns outputs of the form [N*256*256]
class ResNetModelU(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])

        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        # remove the final 2 layers from resnet for deconvolution
        self.res_conv = nn.Sequential(*(list(self.res_conv.children())[:-2]))

        # adding layers required for upsampling
        self.upsample = Upsample(size=(256, 256), mode='bilinear')
        # using convolution layer of kernel=1 beacause that is the easiest way to pool across all channels
        self.conv = nn.Conv2d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.res_conv(inputs)
        x = self.upsample(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        # removing the channels dimension
        x = x.view(x.shape[:1]+x.shape[2:])
        return x


# ResNet model to perform Semantic Segmentation using a decoder architecture (4 transpose convolution layers)
# Returns outputs of the form [N*256*256]
class ResNetModelD(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])

        if pretrained:
            res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        res_modules = list(res_conv.children())
        self.convblock0 = nn.Sequential(*(res_modules[0:4]))
        self.convblock1 = nn.Sequential(*(res_modules[5]))
        self.convblock2 = nn.Sequential(*(res_modules[6]))
        self.convblock3 = nn.Sequential(*(res_modules[7]))

        # defining decoder
        self.decblock0 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 8  -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decblock1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decblock2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decblock3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4),     # 64 -> 256
            nn.BatchNorm2d(1)
        )
        # sigmoid in the end
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # convolution blocks
        x = self.convblock0(inputs)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        # deconvolution blocks
        x = self.decblock0(x)
        x = self.decblock1(x)
        x = self.decblock2(x)
        x = self.decblock3(x)
        x = self.sigmoid(x)
        x = x.view(x.shape[:1]+x.shape[2:])
        return x


# ResNet model to perform Semantic Segmentation using a decoder architecture (4 transpose convolutions) 
# with skip connections
# Returns outputs of the form [N*256*256]
class ResNetModelDS(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])

        if pretrained:
            res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        # TODO: Find a better way to do this
        res_modules = list(res_conv.children())
        self.convblock0 = nn.Sequential(*(res_modules[0:4]))
        self.convblock1 = nn.Sequential(*(res_modules[5]))
        self.convblock2 = nn.Sequential(*(res_modules[6]))
        self.convblock3 = nn.Sequential(*(res_modules[7]))

        # defining decoder
        self.decblock0 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),    # 8  -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decblock1 = nn.Sequential(
            nn.ConvTranspose2d(256*2, 128, kernel_size=2, stride=2),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decblock2 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 64, kernel_size=2, stride=2),   # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decblock3 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 1, kernel_size=4, stride=4),     # 64 -> 256
            nn.BatchNorm2d(1)
        )
        # sigmoid in the end
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # convolution blocks
        x1 = self.convblock0(inputs)
        x2 = self.convblock1(x1)
        x3 = self.convblock2(x2)
        x = self.convblock3(x3)
        # deconvolution blocks with skip connections
        x = self.decblock0(x)
        x = self.decblock1(torch.cat((x, x3), dim=1))
        x = self.decblock2(torch.cat((x, x2), dim=1))
        x = self.decblock3(torch.cat((x, x1), dim=1))
        x = self.sigmoid(x)
        x = x.view(x.shape[:1]+x.shape[2:])
        return x
