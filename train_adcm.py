
import json
from data_preparation import DataHandling 
from model_training import DecayLR
from model_training import ModelTrainer
from data_preparation import LoaderFactory
import torch
from model_maker import get_network


config_file = 'config.json'

with open(config_file, 'r') as f:
    config = json.load(f)
ga_data_dir = config["ga_data_dir"]
fdg_data_dir = config["fdg_data_dir"]
log_dir = config["log_dir"]
output_dir = config["ga_output_dir"]


data_handler = DataHandling(
    ga_data_dir,
    train_mode="NAC",
    target_mode="ADCM"
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
    normalize="suvscale"
    )


# Get the DataLoader for each dataset type
train_loader = loader_factory.get_loader('train', batch_size=4, num_workers=2, shuffle=True)
val_loader = loader_factory.get_loader('val', batch_size=1, num_workers=2, shuffle=False)
test_loader = loader_factory.get_loader('test', batch_size=1, num_workers=2, shuffle=False)


starting_epoch = 0
decay_epoch = 20
# learning_rate = 0.001

learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_network(patch_size = [168, 168, 16], spacing = [4.07, 4.07, 3.00])
# model.load_state_dict(torch.load('/students/2023-2024/master/Shahpouri/LOG/model_3_29_0_30.pth'))
# model.load_state_dict(torch.load('/students/2023-2024/master/Shahpouri/LOG/model_4_1_1_45.pth'))

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

# Assuming 'model' is your DynUNet model instance
# Replace 'nn.ReLU(inplace=True)' with the activation function you want to use.
add_activation_before_output(model, nn.ReLU(inplace=True))

print(model)

model = model.to(device)

loss_function = torch.nn.MSELoss()
l2_lambda = 0.00001  # Regularization strength for L2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=l2_lambda)

max_epochs = 600
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