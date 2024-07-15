
import json
from data_preparation import DataHandling, LoaderFactory
from model_training import ModelTrainer, DecayLR
import torch
from utils import Config
from model_manager import ModelLoader
from torchsummary import summary

config_file = 'config.json'
config = Config(config_file)

# Confirm device setting
print(f"Configured device: {config.device}")


data_handler = DataHandling(
    config.ga_data_dir,
    train_mode=config.train_mode,
    target_mode=config.target_mode,
    external_centers=config.external_centers,
    train_percent=config.train_percent,
    val_percent=config.val_percent,
    test_percent=config.test_percent
    )


train_files = data_handler.get_data_split('train')
val_files = data_handler.get_data_split('val')
test_files = data_handler.get_data_split('test')


loader_factory = LoaderFactory(
    train_files=train_files,
    val_files=val_files,
    test_files=test_files,
    patch_size=config.patch_size,
    spacing=config.spacing,
    spatial_size=config.spatial_size,
    normalize=config.normalize
    )


# Get the DataLoader for each dataset type
train_loader = loader_factory.get_loader('train', batch_size=config.batch_size['train'], num_workers=config.num_workers['train'], shuffle=True)
val_loader = loader_factory.get_loader('val', batch_size=config.batch_size['val'], num_workers=config.num_workers['val'], shuffle=False)
test_loader = loader_factory.get_loader('test', batch_size=config.batch_size['test'], num_workers=config.num_workers['test'], shuffle=False)


model_loader = ModelLoader(config)
model = model_loader.call_model()
print(model)
print(f"Model loaded and moved to device: {config.device}")
summary(model, input_size=(config.in_channels, *config.patch_size))


loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999), weight_decay=config.l2_lambda)
best_metric = float('inf')
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
train_losses = []
val_losses = []

# Define scheduler
lr_lambda = DecayLR(epochs=config.max_epochs, offset=0, decay_epochs=config.decay_epoch).step
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Train loop
trainer = ModelTrainer(model, train_loader, val_loader, optimizer, loss_function, scheduler, config.max_epochs, config.log_dir, config.device, config)
trainer.train()