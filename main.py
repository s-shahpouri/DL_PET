import json
from data_preparation import prepare_data
import torch
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

from data_preparation import prepare_data  # Import from your data preparation script
from model_training import ModelTrainer  # Import from your training script

# Function to read JSON config file
def read_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = prepare_data(config)
    # Initialize model, loss function, and optimizer
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256),
        act=(nn.ReLU6, {"inplace": True}),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Initialize trainer
    trainer = ModelTrainer(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_function, 
        config['max_epochs'], 
        config['val_interval'], 
        config['log_dir'],
        device
    )
    
    trainer.train()
if __name__ == "__main__":
    print("Starting preparing data ...")
    config_path = "/home/shahpouriz/Data/Practic/training_params.json"
    config = read_config(config_path)
    main(config)



    # def log(self):
    #     self.logger.log(f"train set: {len(train_files)}" )
    #     self.logger.log(f"validation set: {len(val_files)}")
    #     self.logger.log(f"max_epochs: {max_epochs}")
    #     self.logger.log(f"val_interval: {val_interval}")
    #     self.logger.log(f"model.channels: {model.channels}")
# trainer.log()
