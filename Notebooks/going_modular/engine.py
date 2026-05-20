"""Contains Training and Testing functions for the model"""

import os 
from tqdm.auto import tqdm
import torch

from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device = None) -> Tuple[float, float]:

        # Putting the device on train mode
        model.train()
          
        # Initiating loss
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(data_loader):
            # Putting data on the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass
            y_pred = model(X)

            # 2. Calculate the loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backwards (Backward Propagation)
            loss.backward()

            # 5. optimizer step (Gradient Descent)
            optimizer.step()

            # Accuracy Metrics
            y_prediction_labels = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_prediction_labels==y).sum().item()/len(y_pred)
        
        train_loss = train_loss / len(data_loader)
        train_acc = train_acc / len(data_loader)
        return train_loss, train_acc

def test_step(model: torch.nn.Module,
         data_loader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device: torch.device) -> Tuple[float, float]:

        # Put the model on eval mode
        model.eval()

        # Initiate test loss and test accuracy
        test_loss, test_acc = 0, 0

        # Inference mode context manager
        with torch.inference_mode():
            for batch, (X, y) in enumerate(data_loader):
                # Put the features and labels on the target device
                X, y = X.to(device), y.to(device)

                # 1. Forward Pass
                test_pred = model(X)

                # 2. Calculate the loss
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()

                # 3. Accuracy Metrics
                test_pred_labels = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
                test_acc += (test_pred_labels==y).sum().item()/len(test_pred)
        
        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)

        return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
        
        # Declare an empty dict to save model performance
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        # Put model on the desired device
        model.to(device)

        # Loop through the training and testing step
        for epoch in tqdm(range(epochs)):
            # train step
            train_loss, train_acc = train_step(model=model,
                                               data_loader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
            
            # test step
            test_loss, test_acc = test_step(model=model,
                                            data_loader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
            
            print(
                f"Epoch: {epoch+1} | "
                f"Train Loss: {train_loss} | "
                f"Train Acc: {train_acc} | "
                f"Test Loss: {test_loss} | "
                f"Test Acc: {test_acc} | "
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
        
        return results

