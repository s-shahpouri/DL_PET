import torch
from utils import get_kernels_strides, add_activation_before_output, CustomSegResNetDS
import torch.nn as nn
from monai.networks.nets import DynUNet

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def call_model(self):
        model_type = self.config.selected_model
        model_func = getattr(self, model_type, self.default_model)
        model = model_func(self.config)

        if self.config.Tuning["enabled"]:
            print("TUNING MODE ...")
            model_path = self.config.Tuning["model_path"]
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"Loaded model from {model_path} for tuning.")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                print("Starting training from scratch.")
        else:
            print(f"Starting a {model_type} model training.")
        
        return model.to(self.config.device)

    def dyn_unet(self, config):
        kernels, strides = get_kernels_strides(patch_size=config.patch_size, spacing=config.spacing)

        model = DynUNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name=config.dynunet["norm_name"],
            deep_supervision=config.dynunet["deep_supervision"],
            deep_supr_num=config.dynunet["deep_supr_num"],
        )
   
        add_activation_before_output(model, nn.ReLU(inplace=True))
        return model

    def default_model(self, config):
        return self.dyn_unet(config)

    def segresnet(self, config):
        model = CustomSegResNetDS(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=config.segresnet["init_filters"],
            norm=config.segresnet["norm"],
            blocks_down=config.segresnet["blocks_down"],
            blocks_up=config.segresnet["blocks_up"],
            dsdepth=config.segresnet["dsdepth"],
            resolution=config.segresnet["resolution"]
        )
        return model
