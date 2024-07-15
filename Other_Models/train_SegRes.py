
import json
from data_preparation import DataHandling 
from model_training import DecayLR
from model_training import ModelTrainer
from data_preparation import LoaderFactory
import torch
from monai.networks.nets import SegResNetDS 
import torch.nn as nn


config_file = 'config.json'

with open(config_file, 'r') as f:
    config = json.load(f)

log_dir = config["log_dir"]

fdg_data_dir = config["fdg_data_dir"]
fdg_output_dir = config['fdg_output_dir']

ga_output_dir = config["ga_output_dir"]
ga_data_dir = config["ga_data_dir"]

artifact_dir = config["artifacts"]
artifact_output_dir = config ["artifact_output_dir"]

artifact_repeated_dir = config["artifact_repeated_dir"]
artifacts_repeated_output_dir = config["artifacts_repeated_output_dir"]


data_handler = DataHandling(
    ga_data_dir,
    train_mode="NAC",
    target_mode="MAC"
    )

train_files = data_handler.get_data_split('train')
val_files = data_handler.get_data_split('val')
test_files = data_handler.get_data_split('test')


loader_factory = LoaderFactory(
    train_files=train_files,
    val_files=val_files,
    test_files=test_files,
    patch_size = [168, 168, 16],
    spacing = [4.07, 4.07, 3.00],
    spatial_size = (168, 168, 320),
    normalize=False
    )


# Get the DataLoader for each dataset type
train_loader = loader_factory.get_loader('train', batch_size=10, num_workers=2, shuffle=True)
val_loader = loader_factory.get_loader('val', batch_size=1, num_workers=2, shuffle=False)
test_loader = loader_factory.get_loader('test', batch_size=1, num_workers=2, shuffle=False)


starting_epoch = 0
decay_epoch = 5
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Instantiate SegResNetDS model
# model = SegResNetDS(
#     spatial_dims=3,
#     init_filters=32,
#     in_channels=1,
#     out_channels=2,
#     act='relu',
#     norm='batch',
#     blocks_down=(1, 2, 2, 4),
#     blocks_up=None,
#     dsdepth=1
# )

class CustomSegResNetDS(SegResNetDS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Modify the final head to include a ReLU activation
        for i, layer in enumerate(self.up_layers):
            head_conv = layer['head']
            relu = nn.ReLU(inplace=True)
            # Create a sequential container with Conv3d followed by ReLU
            self.up_layers[i]['head'] = nn.Sequential(head_conv, relu)

    // "Model": {
    //     "type": "dyn_unet",
        

    // }

# Instantiate SegResNetDS model
model = CustomSegResNetDS(
    spatial_dims=3,
    init_filters=32,
    in_channels=1,
    out_channels=1,
    # act=('leakyrelu', {'negative_slope': 0.01, 'inplace': True}),  # Use LeakyReLU
    norm='instance',  # Use InstanceNorm3d
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    dsdepth=3,  # Set deep supervision depth to 3 for two additional supervision heads
    resolution=[4.07, 4.07, 3.00]  # Handle non-isotropic kernels and strides
)


model = model.to(device)


loss_function = torch.nn.MSELoss()
l2_lambda = 0.00001  # Regularization strength for L2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=l2_lambda)

max_epochs = 300
best_metric = float('inf')
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
train_losses = []
val_losses = []

# Define scheduler
lr_lambda = DecayLR(epochs=max_epochs, offset=0, decay_epochs=decay_epoch).step
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Train loop
trainer = ModelTrainer(model, train_loader, val_loader, optimizer, loss_function, scheduler, max_epochs,log_dir, device)
# trainer.log()
trainer.train()