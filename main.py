import json
import torch
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

from data_preparation2 import DataHandling  # Import from your data preparation script

from model_training import TrainingLogger, ModelTrainer  # Import from your training script
from UNet_model import create_unet

# Function to read JSON config file
def read_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_handling = DataHandling(config)
    loaders, _, _ = data_handling.prepare_data(loaders_to_prepare=["train", "val", "test"])
    val_loader = loaders.get("val")
    train_loader = loaders.get("train")
    # test_loader = loaders.get("test")

    # Initialize model, loss function, and optimizer
    model = create_unet().to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.1) #lr_step_size = 40, lr_gamma = 0.1

    # Initialize trainer
    trainer = ModelTrainer(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_function, 
        scheduler,
        config['max_epochs'], 
        config['val_interval'], 
        config['log_dir'],
        device
    )
    # Write model details
    train_info1 = f"Trian info:\nChannels: {model.channels}\nStrides: {model.strides}\nNum_res_units: {model.num_res_units}\n"
    train_info2 = f"Num_epochs: {config['max_epochs']}\nBatch_size: {config['batch_size']}\nNum_workers: {config['num_workers']}\n"
    trainer.logger.log(train_info1)
    trainer.logger.log(train_info2)

    trainer.train()





if __name__ == "__main__":
    print("Starting preparing data ...")
    config_path = "/homes/zshahpouri/DL_PET/training_params.json"
    config = read_config(config_path)
    main(config)



    # def log(self):
    #     self.logger.log(f"train set: {len(train_files)}" )
    #     self.logger.log(f"validation set: {len(val_files)}")
    #     self.logger.log(f"max_epochs: {max_epochs}")
    #     self.logger.log(f"val_interval: {val_interval}")
    #     self.logger.log(f"model.channels: {model.channels}")
# trainer.log()
