# models.py
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet

def get_kernels_strides(patch_size, spacing):
    """
    Adjusted function to use the correct variable names.
    """
    sizes = patch_size  
    spacings = spacing  
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {patch_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(patch_size, spacing):
    """
    Initializes the DynUNet with dynamically determined kernels and strides.
    """
    kernels, strides = get_kernels_strides(patch_size, spacing)
    print("DyUnet is set:")
    print("Kernel size: ", kernels)
    print("Strides: ", strides)

    net = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="INSTANCE",
        deep_supervision=True,
        deep_supr_num=2,
    )

    # # Traverse the modules of the model and replace the last LeakyReLU with ReLU
    # for name, module in net.named_children():
    #     if name == 'upsamples':
    #         # Assuming 'upsamples' is a nn.ModuleList
    #         for upsample_block in module:
    #             for sub_name, sub_module in upsample_block.named_children():
    #                 if sub_name == 'conv_block':
    #                     for block_name, block_module in sub_module.named_children():
    #                         if isinstance(block_module, nn.LeakyReLU):
    #                             setattr(sub_module, block_name, nn.ReLU(inplace=True))


    return net

##########################################3
import torch.nn as nn
def add_activation_before_output(model, activation_fn):
    """
    Adds an activation function just before the output of the network.
    """
    # Replace the last conv layer with a sequential layer that has conv followed by activation
    old_output_conv = model.output_block.conv.conv
    new_output_block = nn.Sequential(
        old_output_conv,
        activation_fn
    )
    model.output_block.conv.conv = new_output_block

###################################################################
# from monai.networks.nets import DynUNet
# from monai.networks.blocks import UnetOutBlock
# import torch.nn as nn

# import torch.nn as nn

# class CustomOutputBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)):
#         super(CustomOutputBlock, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         return self.activation(self.conv(x))

# # Then, modify the get_network function to use the CustomOutputBlock:
# def get_network(patch_size, spacing):
#     """
#     Initializes the DynUNet with dynamically determined kernels and strides.
#     Adds a ReLU activation function after the final convolutional layer.
#     """
#     kernels, strides = get_kernels_strides(patch_size, spacing)
#     print("DyUnet is set:")
#     print("Kernel size: ", kernels)
#     print("Strides: ", strides)

#     # Initialize the DynUNet as before
#     net = DynUNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=1,
#         kernel_size=kernels,
#         strides=strides,
#         upsample_kernel_size=strides[1:],
#         norm_name="INSTANCE",
#         deep_supervision=True,
#         deep_supr_num=2,
#     )

#     # Modify the final convolutional layer to add a ReLU activation
#     net.output_block = CustomOutputBlock(in_channels=32, out_channels=1)

#     return net

##################################################################
# class DynUNetR(nn.Module):
#     def __init__(self, patch_size, spacing):
#         super().__init__()
#         kernels, strides = get_kernels_strides(patch_size, spacing)
#         self.dynunet = DynUNet(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=1,
#             kernel_size=kernels,
#             strides=strides,
#             upsample_kernel_size=strides[1:],
#             norm_name="INSTANCE",
#             deep_supervision=True,
#             deep_supr_num=2,
#         )
#         self.final_relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.dynunet(x)
#         x = self.final_relu(x)
#         return x


class CustomDynUNet(DynUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add a ReLU activation after the final convolution layer
        self.output_block = nn.Sequential(
            self.output_block,
            nn.ReLU(inplace=True)
        )

