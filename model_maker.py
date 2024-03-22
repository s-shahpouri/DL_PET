# models.py
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet


# def create_unet():
#     return UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=1,
#         channels=(32, 64, 128, 256),
#         act=(nn.ReLU6, {"inplace": True}),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         norm=Norm.BATCH,
#     )


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
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
    )
    return net