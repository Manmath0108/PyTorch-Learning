"""
Trains a PyTorch model using device agnostic code
"""
# 1. Imports
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# 2. Setup Hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# 3. Setup Directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# 4. Set up device 
device = "mps" if torch.mps.is_available else "cpu"

# 5. Create the Transform
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 6. Create DataLoader using data_setup.py 
train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# 7. Create model with help from model_builder.py 
model = model_builder.TinyVGG(
    input_shape = 3,
    hidden_units = HIDDEN_UNITS,
    output_shape = len(class_names)
).to(device)

# 8. Setup Loss Function and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# 9. Start training with help of engine.py 
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# 10. Save the model with help from utils.py 
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
