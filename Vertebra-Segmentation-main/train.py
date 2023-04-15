
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import AttU_Net, NestedUNet, SalsaNext #build_unet
from model_unet import build_unet
from transunet import U_Transformer
from loss import *
from utils import seeding, create_dir, epoch_time

def train(model, loader, optimizer, loss_fn, dis_map, device):
    epoch_loss = 0.0

    model.train()
    if not dis_map:
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    else:
        for x, y, dst in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            dst = dst.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y, dst)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, dis_map, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        if not dis_map:
            for x, y in loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()
        else:
            for x, y, dst in loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                dst = dst.to(device, dtype=torch.float32)

                y_pred = model(x)
                loss = loss_fn(y_pred, y, dst)
                epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def run(loss="DiceBCE"):
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("/content/drive/MyDrive/Boostnet_data/data/training/*"))
    train_y = sorted(glob("/content/drive/MyDrive/Boostnet_data/mask/training/*"))
    train_dist_map = sorted(glob(TRAIN_DIST_MAP_DIR))

    valid_x = sorted(glob("/content/drive/MyDrive/Boostnet_data/data/test/*"))
    valid_y = sorted(glob("/content/drive/MyDrive/Boostnet_data/mask/test/*"))
    valid_dist_map = sorted(glob(VAL_DIST_MAP_DIR))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 1 #2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    if loss == "Boundary" or loss == "BoundaryModified":
        train_dataset = DriveDataset(train_x, train_y, train_dist_map)
        valid_dataset = DriveDataset(valid_x, valid_y, valid_dist_map)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    else:
        train_dataset = DriveDataset(train_x, train_y)
        valid_dataset = DriveDataset(valid_x, valid_y)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

    device = torch.device('cuda')
    model = SalsaNext() #U_Transformer() #
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = None
    dis_map = False
    if loss == "DiceBCE":
        loss_fn = DiceBCELoss()
    elif loss == "Boundary":
        loss_fn = Boundary_Loss()
        dis_map = True
    elif loss == "BoundaryModified":
        loss_fn = Boundary_Loss_Modified()
        dis_map = True

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, dis_map, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, dis_map, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

if __name__ == "__main__":
    run("Boundary")
