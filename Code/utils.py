# Define Dataset that handles both images and fmri data
import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from pathlib import Path
from PIL import Image

import time
import itertools

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

from tqdm import tqdm

import pandas as pd


# Function to load the ROI classes mapping dictionaries
def get_roi_mapping_files(path_to_masks):
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(path_to_masks, r),
            allow_pickle=True).item())


    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(path_to_masks, 
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(path_to_masks, 
            rh_challenge_roi_files[r])))

    roi_names = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
    roi_names.append('All vertices')
    return lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps



class algoDataSet(Dataset):
    

    def __init__(self, data_dir, device, subj_num, hemisphere, roi_name, voxel_subset=False, voxel_subset_num=0):

        self.device = device
        
        # Paths to data
        fmri_path = data_dir + r"/training_fmri/Subj" + str(subj_num)
        imgs_path = data_dir + r"/training_images/Subj" + str(subj_num)
        roi_masks_path = data_dir + r"/roi_masks/Subj" + str(subj_num)
        #voxel_clusterings_path = data_dir + r"/voxel_clusterings/Subj" + str(subj_num)
        
        
        # Get roi masks
        lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps = get_roi_mapping_files(roi_masks_path)
        if (hemisphere=='left'):
            challenge_rois = lh_challenge_rois
        elif (hemisphere=='right'):
            challenge_rois = rh_challenge_rois
        
        # Define image transform
        self.transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
                ])
        # Get image paths
        self.img_paths = np.array(sorted(list(Path(imgs_path).iterdir())))
        

        # Get all fmri data
        if (hemisphere == 'left'):
            all_fmri = np.load(os.path.join(fmri_path, 'subj' + str(subj_num) + '_lh_training_fmri.npy'))
        elif (hemisphere == 'right'):
            all_fmri = np.load(os.path.join(fmri_path, 'subj' + str(subj_num) + '_rh_training_fmri.npy'))
                        
                
        if (roi_name == "all"):
            self.fmri_data = torch.from_numpy(all_fmri)
            del all_fmri
        else:
            # Select fmri data from desired ROI
            roi_group_idx = -1
            roi_idx = -1
            for roi_group in range(len(roi_name_maps)):
                for roi in roi_name_maps[roi_group]:
                    if (roi_name_maps[roi_group][roi] == roi_name):
                        roi_group_idx = roi_group
                        roi_idx = roi
                        break

            # Get indices of voxels in selected ROI
            roi_indices = np.argwhere(challenge_rois[roi_group_idx] == roi_idx)
            self.roi_indices = roi_indices
            # Select fmri data for selected ROI
            fmri_roi = all_fmri[:, roi_indices].reshape(all_fmri.shape[0], roi_indices.shape[0])
            del all_fmri
            self.fmri_data = torch.from_numpy(fmri_roi)
            
        """
        # Get subset of voxels if clustering is selected
        if (voxel_subset==True):
            if (hemisphere=='left'):
                voxel_clusterings_dict = joblib.load(os.path.join(voxel_clusterings_path, "subj" + str(subj_num)
                                                                 + "_lh_voxel_clusterings_dict.joblib"))
            elif (hemisphere=='right'):
                voxel_clusterings_dict = joblib.load(os.path.join(voxel_clusterings_path, "subj" + str(subj_num)
                                                                 + "_rh_voxel_clusterings_dict.joblib"))
            voxel_clusterings = voxel_clusterings_dict[roi_name]
            voxel_cluster = voxel_clusterings[voxel_subset_num]
            self.fmri_data = self.fmri_data[:, voxel_cluster].squeeze() 
        """


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        img = self.transform(img).to(self.device)
        # Load fmri
        fmri = self.fmri_data[idx].to(self.device)
        return fmri, img, idx
    
    def get_img_paths(self):
        return self.img_paths
    



# Function to create dataloaders
def get_dataloaders(data_dir, device, subj_num, hemisphere, roi_name, batch_size=1024, voxel_subset=False, voxel_subset_num=0, use_all_data=False):

    generator1 = torch.Generator().manual_seed(0)
    if (voxel_subset==False):
        dataset = algoDataSet(data_dir, device, subj_num, hemisphere, roi_name)
    else:
        dataset = algoDataSet(data_dir, device, subj_num, hemisphere, roi_name, True, voxel_subset_num)
    # Make sure ROI is not empty
    num_voxels = len(dataset[0][0])
    if (num_voxels==0):
        #print("Empty ROI")
        if (use_all_data):
            return 0,0,0
        else:
            return 0,0,0,0, num_voxels
    if (use_all_data):
        train_size = len(dataset)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, train_size, num_voxels
    else:
        train_size = int(0.85 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, train_size, test_size, num_voxels



# Function to get image-only dataloaders for test predictions
def get_img_dataloader(data_dir, device, subj_num, batch_size=1024):

    dataset = algoDataSet_imgs_only(data_dir, device, subj_num)
    num_images = len(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader, num_images


# Function to get d(validation) dataloader with image paths for visualizations
def get_dataloaders_with_img_paths(data_dir, device, subj_num, hemisphere, roi, batch_size=1024):

    dataset = algoDataSet(data_dir, device, subj_num, hemisphere, roi)
    img_paths = dataset.get_img_paths()
    num_voxels = len(dataset[0][0])
    
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size

    train_idxs = torch.arange(0, train_size)
    test_idxs = torch.arange(train_size, train_size+test_size)

    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    test_dataset = torch.utils.data.Subset(dataset, test_idxs)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    train_img_paths = img_paths[train_idxs]
    test_img_paths = img_paths[test_idxs]
    
    return train_dataloader, test_dataloader, train_size, test_size, num_voxels, train_img_paths, test_img_paths



# Function to create a list of length num_voxels, where each entry is a list containing the ROI(s) (may be none, 1, or multiple) that the voxel is in
# Should pass dataloader containing all voxels as argument
def get_voxel_roi_mappings(data_dir, subj_num, hemisphere, num_voxels):

    roi_masks_path = data_dir + r"/roi_masks/Subj" + str(subj_num)

    # Get roi masks
    lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps = get_roi_mapping_files(roi_masks_path)
    if (hemisphere=='left'):
        challenge_rois = lh_challenge_rois
    elif (hemisphere=='right'):
        challenge_rois = rh_challenge_rois

        
    all_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1",
            "FFA-2", "mTL-faces", "aTL-faces", "OPA",
              "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
           "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    
    roi_indices_dict = {}
    for roi_name in all_rois:
        roi_group_idx = -1
        roi_idx = -1
        for roi_group in range(len(roi_name_maps)):
            for roi in roi_name_maps[roi_group]:
                if (roi_name_maps[roi_group][roi] == roi_name):
                    roi_group_idx = roi_group
                    roi_idx = roi
                    break
        # Get indices of voxels in selected ROI
        roi_indices = np.argwhere(challenge_rois[roi_group_idx] == roi_idx)
        roi_indices_dict[roi_name] = roi_indices 
    
    
    voxel_roi_lists = []
    for voxel_idx in range(num_voxels):
        rois = []
        for roi_name in all_rois:
            if voxel_idx in roi_indices_dict[roi_name]:
                rois.append(roi_name)
        voxel_roi_lists.append(rois)
    return voxel_roi_lists
        
        
# Function to return all fmri data given a dataloader
def get_fmri_from_dataloader(dataloader, num_images, num_voxels):
    fmri = np.zeros((num_images, num_voxels))
    for batch_index, data in enumerate(dataloader):
        fmri_batch = data[0]
        idxs = data[2]
        batch_size = fmri_batch.shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        fmri[low_idx:high_idx] = fmri_batch.cpu().numpy()
        del fmri_batch, idxs
    return fmri
    



def get_linear_models(train_dataloader, cl_model, feature_extractor, num_images, num_voxels, alex_dim, h_dim, 
                      num_pca_comps=500, weighting='linear', num_ks=200):
    
    train_fmri, train_img_h, train_img_features, pca = get_data_for_residual_model(train_dataloader, cl_model, feature_extractor, 
                                                                                   num_images, num_voxels, num_pca_comps, alex_dim, h_dim)
    
    # Get training similarities (note that diagonal is set to 0s so dont have to worry about same-image pairs returning highest similarity
    train_sims = pairwise_cosine_similarity(train_img_h)
    
    # Get similarity-based predicted responses for training images
    train_preds = torch.zeros((num_images, num_voxels))
    print("Computing similarity-based predictions...")
    for img_idx in tqdm(range(num_images)):
        img_sims = train_sims[img_idx, :]
        img_sims_sort_order_ascending = torch.argsort(img_sims)
        total_response = torch.zeros(num_voxels)
        total_weighting = 0
        for k in range(num_ks):
            response_idx = img_sims_sort_order_ascending[-1 * (k+1)]
            response = train_fmri[response_idx]
            if (weighting == 'linear'):
                weight = 1/(k+1)
                total_weighting += weight
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
            elif (weighting == 'square'):
                weight = 1/((k+1)**2)
                total_weighting += weight   
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
            elif (weighting == 'sqrt'):
                weight = 1/((k+1)**0.5)     
                total_weighting += weight   
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
        avg_response = total_response / total_weighting
        train_preds[img_idx] = avg_response
        
    # Compute residuals between train predictions and ground truth
    train_resids = torch.zeros((num_images, num_voxels))
    for img_idx in range(num_images):
        train_resids[img_idx] = torch.add(train_preds[img_idx], -1*train_fmri[img_idx])
        
    # Get PCA features for train images
    train_pca_img_features = pca.transform(train_img_features)
    
    # Train CL model to predict residuals, control model to predict train fmri
    train_fmri = train_fmri.cpu().numpy()
    train_resids = train_resids.cpu().numpy()
    resid_model = LinearRegression().fit(train_pca_img_features, train_resids)
    ctrl_model = LinearRegression().fit(train_pca_img_features, train_fmri)
    
    return resid_model, ctrl_model, train_fmri, train_img_h, pca


def evaluate_linear_models(resid_model, ctrl_model, test_dataloader, train_fmri, train_img_h, pca, cl_model, feature_extractor, num_images, num_voxels, alex_dim, h_dim, 
                      num_pca_comps=500, weighting='linear', num_ks=200):
    
    # Get testing fmri, testing image features, testing image projections
    test_fmri, test_img_h, test_img_features = get_data_for_validation(test_dataloader, cl_model, 
                                                                       feature_extractor, num_images, num_voxels, num_pca_comps, alex_dim, h_dim)
    
    # Get similarity-based predictions using training image projections
    test_preds = torch.zeros((num_images, num_voxels))
    print("Computing similarity-based predictions...")
    # Should have shape (test_images x train_images)
    train_test_img_sims = pairwise_cosine_similarity(test_img_h, train_img_h)
    train_fmri = torch.tensor(train_fmri)
    for img_idx in range(num_images):
        img_sims = train_test_img_sims[img_idx, :]
        img_sims_sort_order_ascending = torch.argsort(img_sims)
        total_response = torch.zeros(num_voxels)
        total_weighting = 0
        for k in range(num_ks):
            response_idx = img_sims_sort_order_ascending[-1 * (k+1)]
            response = train_fmri[response_idx]
            if (weighting == 'linear'):
                weight = 1/(k+1)
                total_weighting += weight
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
            elif (weighting == 'square'):
                weight = 1/((k+1)**2)
                total_weighting += weight   
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
            elif (weighting == 'sqrt'):
                weight = 1/((k+1)**0.5)     
                total_weighting += weight   
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
        avg_response = total_response / total_weighting
        test_preds[img_idx] = avg_response
    
    
    # Get test image pca features
    test_img_pca_features = pca.transform(test_img_features)
    
    # Use trained residual model to predict and add subtract residuals
    pred_resids = resid_model.predict(test_img_pca_features)
    # Subtract predicted residuals to similarity-based predictions to get final CL preds
    cl_preds = np.add(test_preds, -1*pred_resids)
    # Get control predictions
    ctrl_preds = ctrl_model.predict(test_img_pca_features)
    
    # Evaluate correlation accuracies
    ctrl_corr = np.zeros(num_voxels)
    cl_corr = np.zeros(num_voxels)
    for voxel_idx in tqdm(range(num_voxels)):
        ctrl_corr[voxel_idx] = corr(ctrl_preds[:, voxel_idx], test_fmri[:, voxel_idx])[0]
        cl_corr[voxel_idx] = corr(test_preds[:, voxel_idx], test_fmri[:, voxel_idx])[0]
  
    avg_cl_corr = cl_corr.mean()
    avg_ctrl_corr = ctrl_corr.mean()
    print("CL Method Correlation: ", np.round(avg_cl_corr,3))
    print("Control Method Correlation: ", np.round(avg_ctrl_corr,3))
    
    return ctrl_corr, cl_corr, test_fmri, test_preds, pred_resids, cl_preds, ctrl_preds, avg_cl_corr, avg_ctrl_corr



# Function to return fmri data, img_h, pca fit to training images
def get_train_data_for_prediction(device, dataloader, cl_model, feature_extractor, num_images, num_voxels, h_dim, alex_dim):
    
    
    fmri = torch.zeros((num_images, num_voxels))
    img_h = torch.zeros((num_images, h_dim))
    img_alex_features = np.zeros((num_images, alex_dim))
    
    print("Getting training features...")
    for batch_index, data in enumerate(dataloader):
        with torch.no_grad():
            
            fmri_batch = data[0]
            img_batch = data[1]
            batch_idxs = data[2]
            
            batch_size = fmri_batch.shape[0]
            if (batch_index == 0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
                
            fmri[low_idx:high_idx] = fmri_batch
                
            features_batch = feature_extractor(img_batch)
            features_batch_flat = torch.hstack([torch.flatten(l, start_dim=1) for l in features_batch.values()]).cpu().numpy()
            img_alex_features[low_idx:high_idx] = features_batch_flat
            
            _, img_h_batch = cl_model.forward(fmri_batch, img_batch, device, return_h=True)
            img_h[low_idx:high_idx] = img_h_batch
            del fmri_batch, img_batch, _, img_h_batch, features_batch_flat
            
    pca = PCA(n_components=500).fit(img_alex_features)
    img_pca_features = pca.transform(img_alex_features)
            
    return fmri, img_h, img_pca_features, pca


# Function to get image projections, pca features from fit pca from dataloader with fmri and images or just image dataloader
def get_img_features(device, dataloader, is_img_dataloader, cl_model, feature_extractor, pca, num_images, num_voxels, h_dim, alex_dim):
    
    img_h = torch.zeros((num_images, h_dim))
    img_alex_features = np.zeros((num_images, alex_dim))
    
    print("Getting image features...")
    for batch_index, data in enumerate(dataloader):
        with torch.no_grad():
            
            if (is_img_dataloader):
                img_batch = data[0]
            else:
                img_batch = data[1]
            
            batch_size = img_batch.shape[0]
            if (batch_index == 0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
                
            features_batch = feature_extractor(img_batch)
            features_batch_flat = torch.hstack([torch.flatten(l, start_dim=1) for l in features_batch.values()]).cpu().numpy()
            img_alex_features[low_idx:high_idx] = features_batch_flat

            fmri_dummy = torch.zeros((batch_size, num_voxels))
            _, img_h_batch = cl_model.forward(fmri_dummy, img_batch, device, return_h=True)
            img_h[low_idx:high_idx] = img_h_batch
            del img_batch, _, img_h_batch, features_batch, features_batch_flat
            
    img_pca_features = pca.transform(img_alex_features)
    del img_alex_features
            
    return img_h, img_pca_features



# Function to return cl and ctrl preds for test dataloader
def get_test_preds(device, test_dataloader, is_img_dataloader, feature_extractor, train_fmri, train_img_h, train_img_pca_features, cl_model, pca, test_size, num_voxels, h_dim, alex_dim, weighting='linear', num_ks=200):
    
    # Get test img h features, pca features
    test_img_h, test_img_pca_features = get_img_features(device, test_dataloader, is_img_dataloader, cl_model, feature_extractor, pca, test_size, num_voxels, h_dim, alex_dim)
    
    # Get similarity-based predictions
    cl_preds = torch.zeros((test_size, num_voxels))
    print("Computing similarity-based predictions...")
    train_test_img_sims = pairwise_cosine_similarity(test_img_h, train_img_h)
    for img_idx in tqdm(range(test_size)):
        img_sims = train_test_img_sims[img_idx, :]
        img_sims_sort_order_ascending = torch.argsort(img_sims)
        total_response = torch.zeros(num_voxels)
        total_weighting = 0
        for k in range(num_ks):
            response_idx = img_sims_sort_order_ascending[-1 * (k+1)]
            response = train_fmri[response_idx]
            if (weighting == 'linear'):
                weight = 1/(k+1)
                total_weighting += weight
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
            elif (weighting == 'square'):
                weight = 1/((k+1)**2)
                total_weighting += weight   
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
            elif (weighting == 'sqrt'):
                weight = 1/((k+1)**0.5)     
                total_weighting += weight   
                weighted_response = response * weight
                total_response = torch.add(total_response, weighted_response)
        avg_response = total_response / total_weighting
        cl_preds[img_idx, :] = avg_response
        
    cl_preds = cl_preds.numpy()
    
    
    # Fit linear model to control image pca features
    ctrl_model = LinearRegression().fit(train_img_pca_features, train_fmri)
    
    # Get control preds as well to use for voxels with no ROI affiliations or that don't improve with CL
    ctrl_preds = ctrl_model.predict(test_img_pca_features) 
    
    return cl_preds, ctrl_preds




# Function that returns percentage of voxels that were improved by CL method over control method
def get_improved_voxels(ctrl_corrs, cl_corrs):
    
    diff = cl_corrs - ctrl_corrs
    improve_percent = np.count_nonzero(diff > 0) / diff.size
    # Array of size num_voxels with 1 for improved voxels, 0 for not improved voxels
    #improved_voxels = np.zeros(diff.shape[0])
    #improved_voxels[diff > 0] = 1
    return improve_percent, diff



# Fit PCA to training images
def fit_pca(feature_extractor, dataloader, num_images, alex_out_size, batch_size, is_cl_feature_extractor=False, num_voxels=0, device=None):

    print("Fitting PCA...")
    features = np.zeros((num_images, alex_out_size))
    for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # If a CL feature extractor, need to pass dummy fmri data to forward function
        if (is_cl_feature_extractor):
            fmri_dummy = torch.zeros((batch_size, num_voxels)).to(device)
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[0], data[1])
            ft = alex_out_dict['alex.classifier.6'].detach().numpy()

        else:
            with torch.no_grad():
                ft = feature_extractor(data[1])
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()
            
        features[low_idx:high_idx] = ft
        del ft
        
    pca = PCA(n_components=1000).fit(features)
    return pca


# Given already fit PCA, extract image features
def extract_pca_features(feature_extractor, pca, num_images, dataloader, is_img_dataloader, batch_size, is_cl_feature_extractor=False, num_voxels=0, device=None):

    print("Extracting PCA Features...")
    features = np.zeros((num_images, 1000))
    for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        if (is_img_dataloader):
            with torch.no_grad():
                ft = feature_extractor(data[0])
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()

        else:
            # If a CL feature extractor, need to pass dummy fmri data to forward function
            if (is_cl_feature_extractor):
                fmri_dummy = torch.zeros((batch_size, num_voxels)).to(device)
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[0], data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().numpy()
                
            else:
                with torch.no_grad():
                    ft = feature_extractor(data[1])
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()

        # Apply PCA transform
        ft = pca.transform(ft)
        features[low_idx:high_idx] = ft
        del ft
        
    return features