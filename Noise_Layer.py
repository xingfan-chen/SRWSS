import cv2
import torch.nn as nn
import numpy as np
import torch
import kornia
import random
import math


class ScreenShooting(nn.Module):

    def __init__(self):
        super(ScreenShooting, self).__init__()

    def forward(self, embed_image, stylenet):
        noised_image = stylenet(embed_image)
        return noised_image

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, embed_image):
        output = embed_image
        return output