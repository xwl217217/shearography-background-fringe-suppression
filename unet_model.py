""" Full assembly of the parts to form the complete network """

from unet_parts import *
import math

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.conv1 = (Down(64, 16))
        self.conv2 = (Down(16, 1))
        self.outc = (FlattenMLP(128, out_channels=n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.outc(x)
        return logits


def init_weights(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.kaiming_uniform_(model.weight, a=math.sqrt(5), mode='fan_in')
        model.bias.data.fill_(0)
    elif isinstance(model,nn.Conv2d):
        torch.nn.init.kaiming_uniform_(model.weight, a=math.sqrt(5), mode='fan_in')


