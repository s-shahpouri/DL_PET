"""
This module provides essential classes and functions for managing and executing
the training process of a deep learning model, with a focus on medical image analysis.

Classes:
- TrainingLogger: Handles logging of training progress, saving logs to a file with timestamps.
- DecayLR: Manages the learning rate decay during training.
- ModelTrainer: Encapsulates the training and validation loop, handling deep supervision,
  logging, and model saving.

Author: Sama Shahpouri
Last Edit: 25-08-2024
"""

import os
import datetime
from datetime import datetime
import torch
from monai.inferers import sliding_window_inference


class TrainingLogger:
    '''A class to manage logging during training, ensuring logs are saved
    to a file with timestamps.'''

    def __init__(self, directory):
        self.directory = directory
        self.ensure_directory_exists(self.directory)
        self.log_file = self.create_log_file()

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_log_file(self):
        filename = f"{self.directory}/log_{self.get_date()}.txt"
        return open(filename, "w")

    def get_date(self):

        s = datetime.now()
        date = f"{s.month}_{s.day}_{s.hour}_{s.minute}"
        return date

    def log(self, message):
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()  # Flush the output buffer, ensuring immediate write to file


    def close(self):
        self.log_file.close()


class DecayLR:
    """
    A class to manage learning rate decay during training.

    Attributes:
        epochs (int): Total number of training epochs.
        offset (int): Offset to adjust when decay starts.
        decay_epochs (int): Number of epochs during which decay occurs.

    Methods:
        step(epoch):
            Calculates the decay factor for the given epoch.
    """
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (self.epochs - self.decay_epochs)
    

def deep_loss(outputs, target, loss_function, device, weights=None):
    """
    Compute the deep supervision loss for each output feature map.

    Parameters:
    - outputs: Tensor containing all output feature maps, including the final prediction.
    - target: The ground truth tensor.
    - loss_function: The loss function to apply.
    - device: The device on which to perform the calculations.
    - weights: A list of weights for each output's loss. Defaults to equal weighting if None.

    Returns:
    - Weighted average of the computed losses.
    """
    # Unbind the outputs along the first dimension to handle each feature map individually
    output_maps = torch.unbind(outputs, dim=1)
    
    if weights is None:
        # If no weights specified, use equal weights
        weights = [1.0 / len(output_maps)] * len(output_maps)
    elif sum(weights) != 1:
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]

    total_loss = 0.0
    for output, weight in zip(output_maps, weights):
        # Resize target to match the output size if necessary
        resized_target = torch.nn.functional.interpolate(target, size=output.shape[2:], mode='nearest').to(device)
        # Compute loss for the current output
        loss = loss_function(output, resized_target)
        # Accumulate weighted loss
        total_loss += weight * loss

    return total_loss


def deep_loss2(outputs, target, loss_function, device, weights=None):
    """
    Compute the deep supervision loss for each output feature map.

    Parameters:
    - outputs: List of tensors containing all output feature maps, including the final prediction.
    - target: The ground truth tensor.
    - loss_function: The loss function to apply.
    - device: The device on which to perform the calculations.
    - weights: A list of weights for each output's loss. Defaults to equal weighting if None.

    Returns:
    - Weighted average of the computed losses.
    """
    if weights is None:
        weights = [1.0 / len(outputs)] * len(outputs)
    elif sum(weights) != 1:
        total = sum(weights)
        weights = [w / total for w in weights]

    total_loss = 0.0
    for output, weight in zip(outputs, weights):
        resized_target = torch.nn.functional.interpolate(target, size=output.shape[2:], mode='nearest').to(device)
        loss = loss_function(output, resized_target)
        total_loss += weight * loss

    return total_loss


class ModelTrainer:
    """
    A class to handle training, validation, and saving of a deep learning model, along with logging.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        loss_function (callable): Loss function used during training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        max_epochs (int): Maximum number of training epochs.
        log_dir (str): Directory where logs and models are saved.
        device (torch.device): The device (CPU or CUDA) for computation.
        config (object): Configuration object containing training parameters.

    Methods:
        log_config():
            Logs the configuration parameters used for training.
        
        train():
            Executes the training and validation loop, logging results and saving the best model.
        
        save_model():
            Saves the current model state to the log directory.
    """
    def __init__(self, model, train_loader, val_loader,
                 optimizer, loss_function, scheduler, max_epochs,
                 log_dir, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler  # Add scheduler to the class initialization
        self.max_epochs = max_epochs
        self.val_interval = 2
        self.directory = log_dir
        self.device = device 
        self.logger = TrainingLogger(log_dir)
        self.best_metric = float('inf')
        self.best_metric_epoch = -1
        self.config = config  # Save the configuration

        self.log_config()  # Log the configuration parameters

    def log_config(self):
        # Log important parameters from the config
        self.logger.log("Training Configuration Parameters:")
        self.logger.log(f"Device: {self.config.device}")
        self.logger.log(f"Train mode: {self.config.train_mode}")
        self.logger.log(f"Target mode: {self.config.target_mode}")
        self.logger.log(f"Learning rate: {self.config.learning_rate}")
        self.logger.log(f"Max epochs: {self.config.max_epochs}")
        self.logger.log(f"Decay epoch: {self.config.decay_epoch}")
        self.logger.log(f"Patch size: {self.config.patch_size}")
        self.logger.log(f"Spacing: {self.config.spacing}")
        self.logger.log(f"Batch size (train): {self.config.batch_size['train']}")
        self.logger.log(f"Batch size (val): {self.config.batch_size['val']}")
        self.logger.log(f"Batch size (test): {self.config.batch_size['test']}")
        self.logger.log(f"Tuning enabled: {self.config.Tuning['enabled']}")
        self.logger.log(f"Tuning model path: {self.config.Tuning['model_path']}")
        self.logger.log(f"Selected model: {self.config.selected_model}")

    def train(self):
        
        for epoch in range(self.max_epochs):
            self.logger.log("-" * 10)
            self.logger.log(f"epoch {epoch + 1}/{self.max_epochs}")

            self.model.train()
            epoch_loss = 0
            step = 0

            for batch_data in self.train_loader:
                step += 1
                inputs, targets = batch_data["image"].to(self.device), batch_data["target"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Check if deep supervision is used
                if isinstance(outputs, tuple) or (outputs.dim() > targets.dim()):
                    # Outputs from deep supervision
                    loss = deep_loss(outputs, targets, self.loss_function, self.device)
                elif isinstance(outputs, list): 
                    loss = deep_loss2(outputs, targets, self.loss_function, self.device)
                else:
                    # Standard output handling
                    outputs = torch.squeeze(outputs)
                    targets = torch.squeeze(targets, dim=1)  # Adjust for channel dimension if necessary
                    loss = self.loss_function(outputs, targets)
                    print('deep loss does not work!')
                

                l1_lambda = 0.0001  # Regularization strength for L1
                # L1 regularization
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.logger.log(f"{step}/{len(self.train_loader.dataset) // self.train_loader.batch_size}, train_loss: {loss.item():.4f}")

            epoch_loss /= step
            self.logger.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # Step the scheduler here, after the training phase and before the validation phase
            self.scheduler.step()
            self.logger.log(f"current lr: {self.scheduler.get_last_lr()[0]}")

            # Validation logic remains largely the same
            if (epoch + 1) % self.val_interval == 0:
                self.model.eval()
                val_loss = 0
                roi_size = (168, 168, 32)
                sw_batch_size = 16
                
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs, val_targets = val_data["image"].to(self.device), val_data["target"].to(self.device)

                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, self.model)
                        if isinstance(val_outputs, list):
                            val_loss += deep_loss2(val_outputs, val_targets, self.loss_function, self.device).item()
                        else:
                            val_loss += self.loss_function(val_outputs, val_targets).item()

                        
                        # val_loss += self.loss_function(val_outputs, val_targets).item()

                val_loss /= len(self.val_loader)
                self.logger.log(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self.best_metric_epoch = epoch + 1
                    self.save_model()

        self.logger.close()

    def save_model(self):
        model_filename = f"model_{self.logger.get_date()}.pth"
        torch.save(self.model.state_dict(), os.path.join(self.directory, model_filename))
        self.logger.log(f"Saved {model_filename} model, best_metric: {self.best_metric:.4f}, epoch: {self.best_metric_epoch}")

