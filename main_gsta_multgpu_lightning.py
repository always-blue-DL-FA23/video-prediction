import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import datetime
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

# Assuming nn_models and unet_models are modules you have that can be imported.
from nn_models import VideoDataset, SimVPgsta

def datetime_formatted():
    # Get current date and time
    now = datetime.datetime.now()
    # Format the datetime as a string in the specified forma
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    return str(formatted_now)

class MyModel(pl.LightningModule):
    def __init__(self, shape_in, learning_rate=0.001):
        super().__init__()
        self.model = SimVPgsta(shape_in=shape_in)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.start_time = 0

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        # Record the start time of the epoch
        self.start_time = time.time()

    def on_epoch_end(self):
        # Calculate and log the epoch duration
        epoch_duration = time.time() - self.start_time
        self.log('epoch_duration', epoch_duration)

    def training_step(self, batch, batch_idx, dataset_type='labeled'):
        images, _ = batch
        input_frames = images[:, :11].to(self.device)
        target_frame = images[:, 21].to(self.device)
        predicted_frames = self(input_frames)
        predicted_target_frame = predicted_frames[:, -1]
        loss = self.criterion(predicted_target_frame, target_frame)
        self.log(f'{dataset_type}_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        input_frames = images[:, :11].to(self.device)
        target_frame = images[:, 21].to(self.device)
        predicted_frames = self(input_frames)
        predicted_target_frame = predicted_frames[:, -1]
        loss = self.criterion(predicted_target_frame, target_frame)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        # Manually iterate over the unlabeled dataset
        unlabeled_loader = self.trainer.datamodule.unlabeled_dataloader()
        for batch in unlabeled_loader:
            self.training_step(batch, None, dataset_type='unlabeled')

class MyDataModule(pl.LightningDataModule):
    def __init__(self, base_path, transform, batch_size=16):
        super().__init__()
        self.base_path = base_path
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(self.base_path, dataset_type='train', transform=self.transform)
        self.val_dataset = VideoDataset(self.base_path, dataset_type='val', transform=self.transform)
        self.unlabeled_dataset = VideoDataset(self.base_path, dataset_type='unlabeled', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def unlabeled_dataloader(self):
        return DataLoader(self.unlabeled_dataset, batch_size=self.batch_size, shuffle=True)

# logging
logname = '../outs/logs/vp_'+str(datetime_formatted())+'.log'
logging.basicConfig(filename=logname, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

stime = datetime_formatted()
logging.info("Logging beginning at "+str(stime))
print("Logging beginning at "+str(stime))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_gpus = torch.cuda.device_count()
logging.info(f"Using device: {device}")
logging.info(f"# device: {num_gpus}")


# Transforms
transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[0.0571, 0.0567, 0.0614])
])

# Data module
base_path = '/scratch/sa7445/data/dataset'  # Adjust as needed
data_module = MyDataModule(base_path, transform)

# Model
shape_in = (11, 3, 160, 240)  # Adjust based on your data
model = MyModel(shape_in=shape_in)

# Logger and checkpoint callback
logger = TensorBoardLogger("tb_logs", name="my_model")
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss")

# Trainer
trainer = pl.Trainer(max_epochs=1, logger=logger, callbacks=[checkpoint_callback],accelerator="cuda", devices=4, strategy="fsdp",accumulate_grad_batches=4)

# Train and validate the model
trainer.fit(model, data_module)
trainer.validate(model, data_module)

