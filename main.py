import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.pyplot as plt
from torchvision import transforms

#local imports
import model_errors
from nn_models import VideoDataset, SimVP

transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other transformations here
])
base_path = '../dataset'
train_dataset = VideoDataset(base_path, dataset_type='train', transform=transform)
val_dataset = VideoDataset(base_path, dataset_type='val', transform=transform)
unlabeled_dataset = VideoDataset(base_path, dataset_type='unlabeled', transform=transform)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


#training
epochs=10
shape_in = (11, 3, 128, 128)  # You need to adjust these dimensions based on your actual data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = SimVP(shape_in=shape_in).to(device)
model.train()

frame_prediction_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# OneCycleLR Scheduler
total_steps = epochs * len(train_loader)  # Total number of training steps
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)

print("before")

for epoch in range(epochs):
    for batch in train_loader:
        images, _ = batch
        input_frames = images[:, :11].to(device)
        target_frame = images[:, 21].to(device)

        # Forward pass
        predicted_frames = model(input_frames)
        predicted_target_frame = predicted_frames[:, -1]

        # Loss computation
        loss = frame_prediction_criterion(predicted_target_frame, target_frame)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")

model_save_path = 'my_model.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)

# Inform the user
print(f'Model saved to {model_save_path}')

model.eval()
model.to(device)
mse_loss = nn.MSELoss()
total_loss = 0.0
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:
        images, _ = batch
        input_frames = images[:, :11].to(device)  # First 11 frames
        actual_22nd_frame = images[:, 21].to(device)
        # Forward pass to get the predictions
        predicted_frames = model(input_frames)
        predicted_22nd_frame = predicted_frames[:, -1]
        loss = mse_loss(predicted_22nd_frame, actual_22nd_frame)
        total_loss += loss.item()

# Calculate the average loss
average_loss = total_loss / len(validation_loader)
print(f"Average MSE Loss on the validation dataset: {average_loss}")

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

# Calculate the average loss
average_loss = total_loss / len(validation_loader)
print(f"Average MSE Loss on the validation dataset: {average_loss}")

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
plt.savefig('my_plot.png') 