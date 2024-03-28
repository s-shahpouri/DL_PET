import os
import datetime
from datetime import datetime
import torch
from monai.networks.nets import DynUNet
from torch import nn
from monai.inferers import sliding_window_inference

class TrainingLogger:
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


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, loss_function, scheduler, max_epochs,
                 log_dir, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler  # Add scheduler to the class initialization
        self.max_epochs = max_epochs
        self.val_interval = 2
        self.directory = log_dir
        self.device = device  # Assuming device is passed as a parameter
        self.logger = TrainingLogger(log_dir)
        self.best_metric = float('inf')
        self.best_metric_epoch = -1


    # def log(self):
    #     self.logger.log(f"train set: {len(train_files)}" )
    #     self.logger.log(f"validation set: {len(val_files)}")
    #     self.logger.log(f"max_epochs: {max_epochs}")
    #     self.logger.log(f"model.filters: {model.filters}")



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
                else:
                    # Standard output handling
                    outputs = torch.squeeze(outputs)
                    targets = torch.squeeze(targets, dim=1)  # Adjust for channel dimension if necessary
                    loss = self.loss_function(outputs, targets)
                
# # L1 Regularization
#         l1_reg = torch.tensor(0., requires_grad=True).to(device)
#         for name, param in model.named_parameters():
#             l1_reg = l1_reg + torch.norm(param, 1)
        
#         loss += lambda_reg * l1_reg


                # loss = self.loss_function(outputs, targets)
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

                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                        val_loss += self.loss_function(val_outputs, val_targets).item()

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

