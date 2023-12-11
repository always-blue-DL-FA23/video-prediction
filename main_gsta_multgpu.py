import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torch.optim as optim
import datetime
import time
import json
import logging
import numpy as np

import torchmetrics

import matplotlib.pyplot as plt
from torchvision import transforms

#local imports
# import model_errors
from nn_models import VideoDataset, SimVP, HiddenVideoDataset, SimVPTAU, SimVPgsta
from unet_models import ImageDataset, ImageDatasettrainunet, UNet

# read config
with open('config.json', 'r') as file:
    configs = json.load(file)
# print(configs['vp_epochs'])
# print(configs['unet_epochs'])


def datetime_formatted():
    # Get current date and time
    now = datetime.datetime.now()
    # Format the datetime as a string in the specified forma
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    return str(formatted_now)

# logging
logname = '../outs/logs/vp_'+str(datetime_formatted())+'.log'
logging.basicConfig(filename=logname, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

stime = datetime_formatted()
logging.info("Logging beginning at "+str(stime))
print("Logging beginning at "+str(stime))

transform = transforms.Compose([
    # transforms.Resize((height, width)), # Specify your desired height and width
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(), # You can specify parameters like brightness, contrast, etc.
    transforms.ToTensor(),
    # transforms.Normalize(mean, std) # Specify the mean and std for your dataset
])
#base_path = '../dataset'
base_path = '/scratch/sa7445/data/dataset'

train_dataset = VideoDataset(base_path, dataset_type='train', transform=transform)
val_dataset = VideoDataset(base_path, dataset_type='val', transform=transform)
unlabeled_dataset = VideoDataset(base_path, dataset_type='unlabeled', transform=transform)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset,batch_size=16,shuffle=True)


#training
epochs=30
shape_in = (11, 3, 160, 240)  # You need to adjust these dimensions based on your actual data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_gpus = torch.cuda.device_count()
logging.info(f"Using device: {device}")
logging.info(f"# device: {num_gpus}")


# Initialize the model
model = SimVPgsta(shape_in=shape_in)
nn.DataParallel(model).to(device) if num_gpus > 1 else model.to(device)
model.train()

frame_prediction_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# OneCycleLR Scheduler
total_steps = epochs * (len(train_loader)+len(unlabeled_loader)) *2  # Total number of training steps
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)

print("before training")
logging.info("This is an info message")


# for epoch in range(int(configs['vp_epochs']):
for epoch in range(epochs):
    epoch_loss = 0.0  # Initialize epoch loss
    num_batches = 0
    start_time_unlabeled = time.time()
    # first train on unlabeled dataset
    for batch in unlabeled_loader:
        images, _ = batch
        
        input_frames = images[:, :11].to(device)
        target_frame = images[:, 21].to(device)
        optimizer.zero_grad()
        # Forward pass
        predicted_frames = model(input_frames)
        predicted_target_frame = predicted_frames[:, -1]

        # Loss computation
        loss = frame_prediction_criterion(predicted_target_frame, target_frame)
        epoch_loss += loss.item()  # Accumulate loss
        num_batches += 1
        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        # print(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
        logging.info(f"Epoch Unlabeled [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Batch Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
    avg_epoch_loss = epoch_loss / num_batches
    end_time_unlabeled = time.time()  # End time of the epoch
    epoch_duration_unlabeled = end_time_unlabeled - start_time_unlabeled
    logging.info(f"Epoch [{epoch+1}/{epochs}], Average Loss Epoch: {avg_epoch_loss}, Duration: {epoch_duration_unlabeled:.2f} seconds, LR: {scheduler.get_last_lr()[0]}")


for epoch in range(epochs):
    # now train on training dataset
    start_time = time.time()
    epoch_loss_train = 0.0  # Initialize epoch loss
    num_batches_train = 0
    for batch in train_loader:
        images, _ = batch
        input_frames = images[:, :11].to(device)
        target_frame = images[:, 21].to(device)

        # Forward pass
        predicted_frames = model(input_frames)
        predicted_target_frame = predicted_frames[:, -1]

        # Loss computation
        loss = frame_prediction_criterion(predicted_target_frame, target_frame)
        epoch_loss_train += loss.item()  # Accumulate loss
        num_batches_train += 1
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        # print(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
        logging.info(f"Train Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Batch Training Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
    avg_epoch_loss_train = epoch_loss_train / num_batches_train
    end_time = time.time()  # End time of the epoch
    epoch_duration = end_time - start_time
    logging.info(f"Epoch [{epoch+1}/{epochs}], Average Loss Epoch Train: {avg_epoch_loss_train}, Duration: {epoch_duration:.2f} seconds, LR: {scheduler.get_last_lr()[0]}")

model_save_path = '../outs/models/my_model_' +str(datetime_formatted())+'.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)

# Inform the user
print(f'Model saved to {model_save_path}')
logging.info(f'Model saved to {model_save_path}')


batch = next(iter(val_loader))
input_frames, _ = batch
input_frames = input_frames.to(device)

# Predict the 22nd frame
model.eval()
with torch.no_grad():
    predicted_frames = model(input_frames[:, :11])  # Use first 11 frames as input
    predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction

# Move tensors to CPU for plotting
predicted_22nd_frame = predicted_22nd_frame.cpu()
actual_22nd_frame = input_frames[:, 21].cpu()  # Actual 22nd frame

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0)  # Change dimensions from CxHxW to HxWxC
    tensor = tensor.numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
    return tensor

# Convert tensors to images
predicted_image = tensor_to_image(predicted_22nd_frame[0])  # First sample in the batch
actual_image = tensor_to_image(actual_22nd_frame[0])  # First sample in the batch

# Plot the images for comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(predicted_image)
plt.title('Predicted 22nd Frame')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(actual_image)
plt.title('Actual 22nd Frame')
plt.axis('off')

# plt.show()
plt.savefig('../outs/images/diff_plot_'+datetime_formatted()+'.png') 
