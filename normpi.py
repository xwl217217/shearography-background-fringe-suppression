import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt


class normpi(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        pitensor = tensor * 2 * torch.pi - torch.pi
        return pitensor
