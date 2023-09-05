import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from torchmetrics.functional import pairwise_cosine_similarity

import torchextractor as tx

from tqdm import tqdm


# Define CL loss function
def cl_loss(z_f, z_i, temp):
    # Shape of z_f and z_i is (batch_size x dim_z)
    sims = pairwise_cosine_similarity(z_f, z_i)
    exp_sims = torch.exp(sims / temp)
    positive_sims = torch.diagonal(exp_sims)
    negative_sims = torch.sum(exp_sims, dim=0)
    loss = -torch.log(positive_sims / negative_sims).mean()
    return loss

# Define CL model
class CLR_model(torch.nn.Module):

    def __init__(self, num_voxels, h_dim, z_dim, alex_frozen=False):

        super(CLR_model, self).__init__()

        # Alexnet encoding layers for images
        self.alex = models.alexnet(weights='DEFAULT')
        
        # Freeze alexnet layers if selected
        if (alex_frozen):
            for param in self.alex.parameters():
                param.requires_grad = False

        # Linear mappings to make alex_h and fmri_h same dim
        self.alex_h = nn.Linear(1000, h_dim, bias=False)
        self.fmri_h = nn.Linear(num_voxels, h_dim, bias=False)

        # MLP projection head
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=True),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim, bias=True),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim, bias=False))

    def forward(self, fmri, image, return_h=False, return_alex_intermediate=False):
        # Get output from alexnet layers
        alex_out = self.alex(image)
        # Get output from h
        img_h = self.alex_h(alex_out)
        fmri_h = self.fmri_h(fmri)
        if (return_h):
            return fmri_h, img_h
        else:
            img_z = self.mlp(img_h)
            fmri_z = self.mlp(fmri_h)
            # Return output from MLP projection head for fmri and image, alexnet features of image
            return fmri_z, img_z

    def get_proj(self, fmri, image):
        alex_out = self.alex(image)
        # Get output from h (projections)
        img_h = self.alex_h(alex_out)
        fmri_h = self.fmri_h(fmri)
        return fmri_h, img_h
    
    
# Helper functions for traning

def train_one_step(model, data, optimizer, temp, cross_subj, device=None):
    optimizer.zero_grad()
    # Get output of projection head
    if (cross_subj):
        fmri_z, img_z = model(data[0], data[1], device)
    else:
        fmri_z, img_z = model(data[0], data[1])
    # Compute loss
    loss = cl_loss(fmri_z, img_z, temp)
    # Update weights
    loss.backward()
    optimizer.step()
    return loss

def val_one_step(model, data, temp, cross_subj=False, device=None):
    # Get output of projection head
    if (cross_subj):
        fmri_z, img_z = model(data[0], data[1], device)
    else:
        fmri_z, img_z = model(data[0], data[1])
    # Compute loss
    loss = cl_loss(fmri_z, img_z, temp)
    return loss

def train_one_epoch(model, device, dataloader, optimizer, scheduler, temp, cross_subj):
    model.train()
    total_loss = torch.tensor(0 ,dtype=torch.float64, device=device)
    for batch_index, data in enumerate(dataloader):
        loss = train_one_step(model, data, optimizer, temp, cross_subj, device)
        total_loss += loss
    return total_loss

def val_loss(model, dataloader, temp, cross_subj, device=None):
    model.eval()
    total_loss = 0
    for batch_index, data in enumerate(dataloader):
        with torch.no_grad():
            loss = val_one_step(model, data, temp, cross_subj, device)
            total_loss += loss
    return total_loss



# Main training function

def train(model, device, train_dataloader, test_dataloader, optimizer, scheduler, epochs, temp, cross_subj=False):

    epoch = 0
    for epoch in tqdm(range(epochs)):
        total_train_loss = train_one_epoch(model, device, train_dataloader, optimizer, scheduler, temp, cross_subj)
        total_val_loss = val_loss(model, test_dataloader, temp, cross_subj, device)
        scheduler.step(total_val_loss)
        print("Epoch " + str(epoch) + " train loss: " + str(total_train_loss.item()))
        print("Epoch " + str(epoch) + " val loss: " + str(total_val_loss.item()))

    return model


# Function to get CL model, optimizer and scheduler
def get_CL_model(num_voxels, device, lr=0.0001, alex_frozen=False):
    h_dim = int(num_voxels*0.8)
    z_dim = int(num_voxels*0.2)
    model = CLR_model(num_voxels, h_dim, z_dim, alex_frozen)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return model, optimizer, scheduler



# Define CL model using cross-subject alexnet layers
class CLR_model_cross_subj(torch.nn.Module):

    def __init__(self, alexnet_feature_extractor, num_voxels_train_subj, num_voxels_target, h_dim, z_dim):

        super(CLR_model_cross_subj, self).__init__()
        
        # Alexnet feature extractor from other subject
        self.alex = alexnet_feature_extractor
        self.num_voxels_trained_subj = num_voxels_train_subj
        # Freeze alexnet layers
        for param in self.alex.parameters():
            param.requires_grad = False

        # Linear mappings to make alex_h and fmri_h same dim
        self.alex_h = nn.Linear(1000, h_dim, bias=False)
        self.fmri_h = nn.Linear(num_voxels_target, h_dim, bias=False)

        # MLP projection head
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=True),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim, bias=True),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim, bias=False))

    def forward(self, fmri, image, device, return_h=False):
        # Get output from alexnet layers
        batch_size = fmri.shape[0]
        # CL model requires fmri and img as input, so use dummy fmri data with same shape
        fmri_dummy = torch.zeros((batch_size, self.num_voxels_trained_subj)).to(device)
        _, alex_out_dict = self.alex(fmri_dummy, image)
        alex_out = alex_out_dict['alex.classifier.6']
        # Get output from h
        img_h = self.alex_h(alex_out)
        del _, alex_out_dict, alex_out
        fmri_h = self.fmri_h(fmri)
        if (return_h):
            return fmri_h, img_h
        else:
            img_z = self.mlp(img_h)
            fmri_z = self.mlp(fmri_h)
            # Return output from MLP projection head for fmri and image, alexnet features of image
            return fmri_z, img_z

    def get_proj(self, fmri, image):
        alex_out = self.alex(image)
        # Get output from h (projections)
        img_h = self.alex_h(alex_out)
        fmri_h = self.fmri_h(fmri)
        return fmri_h, img_h




# Function to get model that uses other subject's trained alexnet layers (frozen)
def get_cross_subject_CL_model(trained_model, num_voxels_trained_subj, num_voxels_target, device, lr=0.0001):
    
    # Create feature extractor from trained subject's model (output of Alexnet)
    trained_subj_alexnet_extractor = tx.Extractor(trained_model, ["alex.classifier.6"])
    
    # Specifics for MLP projection head of target model
    h_dim = int(num_voxels_target*0.8)
    z_dim = int(num_voxels_target*0.2)
    
    model = CLR_model_cross_subj(trained_subj_alexnet_extractor, num_voxels_trained_subj, num_voxels_target, h_dim, z_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return model, optimizer, scheduler




# Class for baseline regression network
class fmri_reg(torch.nn.Module):
    def __init__(self, num_voxels):
        super(fmri_reg, self).__init__()
        # Alexnet layers
        self.alex = models.alexnet(weights='DEFAULT')
        self.num_voxels = num_voxels
        # Projection to voxels
        self.proj = nn.Linear(1000, num_voxels, bias=True)
    def forward(self, image):
        out = self.alex(image)
        out = self.proj(out)
        return out
    
    
# Helper function for training regression network
def train_one_step_reg(model, data, optimizer, loss_func):
    optimizer.zero_grad()
    # Get output of forward pass
    fmri_pred = model(data[1])
    fmri_true = data[0]
    # Compute loss
    loss = loss_func(fmri_pred, fmri_true)
    # Update weights
    loss.backward()
    optimizer.step()
    return loss


def val_one_step_reg(model, data, loss_func):
    # Get output of forward pass
    fmri_pred = model(data[1])
    fmri_true = data[0]
    # Compute loss
    loss = loss_func(fmri_pred, fmri_true)
    return loss


def train_one_epoch_reg(model, device, dataloader, optimizer, loss_func):
    model.train()
    total_loss = torch.tensor(0, dtype=torch.float64, device=device)
    for batch_index, data in enumerate(dataloader):
        loss = train_one_step_reg(model, data, optimizer, loss_func)
        total_loss += loss
    return total_loss


def val_loss_reg(model, device, dataloader, loss_func):
    model.eval()
    total_loss = torch.tensor(0, dtype=torch.float64, device=device)
    for batch_index, data in enumerate(dataloader):
        with torch.no_grad():
            loss = val_one_step_reg(model, data, loss_func)
            total_loss += loss
    return total_loss


# Main training function
def train_reg(model, device, train_dataloader, test_dataloader, optimizer, epochs):
    epoch = 0
    loss_func = torch.nn.MSELoss()
    for epoch in tqdm(range(epochs)):
        total_train_loss = train_one_epoch_reg(model, device, train_dataloader, optimizer, loss_func)
        total_val_loss = val_loss_reg(model, device, test_dataloader, loss_func)
        print("Epoch " + str(epoch) + " train loss: " + str(total_train_loss.item()))
        print("Epoch " + str(epoch) + " val loss: " + str(total_val_loss.item()))
    return model


# Function to set up regression network
def get_reg_model(num_voxels, device, lr=0.0001):
    
    model = fmri_reg(num_voxels)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return model, optimizer
