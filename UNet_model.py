# models.py
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def create_unet():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256),
        act=(nn.ReLU6, {"inplace": True}),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
