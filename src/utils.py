
import joblib
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from torchmetrics.functional import pairwise_cosine_similarity
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import itertools
import time
from PIL import Image
from pathlib import Path
import numpy as np
import random
import os
os.environ["OMP_NUM_THREADS"] = '1'


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
            if r2[0] != 0:  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
    roi_names.append('All vertices')
    return lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps


# Function to define fmri and images dataset
class algoDataSet(Dataset):

    def __init__(self, data_dir, device, subj_num, hemisphere, roi_name):

        self.device = device

        # Paths to data
        fmri_path = os.path.join(
            data_dir, "training_fmri", "Subj" + str(subj_num))
        imgs_path = os.path.join(
            data_dir, "training_images", "Subj" + str(subj_num))
        roi_masks_path = os.path.join(
            data_dir, "roi_masks", "Subj" + str(subj_num))

        # Get roi masks
        lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps = get_roi_mapping_files(
            roi_masks_path)
        if hemisphere == 'left':
            challenge_rois = lh_challenge_rois
        elif hemisphere == 'right':
            challenge_rois = rh_challenge_rois

        # Define image transform
        self.transform = transforms.Compose([
            # resize the images to 224x24 pixels
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # convert the images to a PyTorch tensor
            # normalize the images color channels
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Get image paths
        self.img_paths = np.array(sorted(list(Path(imgs_path).iterdir())))

        # Get all fmri data
        if hemisphere == 'left':
            all_fmri = np.load(os.path.join(
                fmri_path, 'subj' + str(subj_num) + '_lh_training_fmri.npy'))
        elif hemisphere == 'right':
            all_fmri = np.load(os.path.join(
                fmri_path, 'subj' + str(subj_num) + '_rh_training_fmri.npy'))
        if roi_name == 'all':
            self.fmri_data = torch.from_numpy(all_fmri)
            del all_fmri
        else:
            # Select fmri data from desired ROI
            roi_group_idx = -1
            roi_idx = -1
            for roi_group in range(len(roi_name_maps)):
                for roi in roi_name_maps[roi_group]:
                    if roi_name_maps[roi_group][roi] == roi_name:
                        roi_group_idx = roi_group
                        roi_idx = roi
                        break
            # Get indices of voxels in selected ROI
            roi_indices = np.argwhere(challenge_rois[roi_group_idx] == roi_idx)
            self.roi_indices = roi_indices
            # Select fmri data for selected ROI
            fmri_roi = all_fmri[:, roi_indices].reshape(
                all_fmri.shape[0], roi_indices.shape[0])
            del all_fmri
            self.fmri_data = torch.from_numpy(fmri_roi)

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
def get_dataloaders(data_dir, device, subj_num, hemisphere, roi_name, batch_size=1024, use_all_data=False, shuffle=True):

    # Seed generator
    torch.manual_seed(0)

    # Create dataset for fmri and image data
    dataset = algoDataSet(data_dir, device, subj_num, hemisphere, roi_name)
    # Make sure ROI is not empty
    num_voxels = len(dataset[0][0])
    if num_voxels == 0:
        print("Empty ROI")
        if use_all_data:
            return 0, 0, 0
        else:
            return 0, 0, 0, 0, num_voxels
    if use_all_data:
        train_size = len(dataset)
        train_dataloader = DataLoader(dataset, batch_size=batch_size)
        return train_dataloader, train_size, num_voxels
    else:
        train_size = int(0.85 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader, train_size, test_size, num_voxels


# Function to create dataloaders for 5-fold cross validation (splitting training set into train/val)
# Fold number can be 0, 1, 2, 3, or 4
def get_dataloaders_cv(data_dir, device, subj_num, hemisphere, roi_name, batch_size=1024, fold_num=0):

    # Seed generator
    torch.manual_seed(0)

    # Create dataset for fmri and image data
    dataset = algoDataSet(data_dir, device, subj_num, hemisphere, roi_name)

    # Make sure ROI is not empty
    num_voxels = len(dataset[0][0])
    if num_voxels == 0:
        return 0, 0, 0, 0, num_voxels

    else:

        train_size = int(0.75 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        train_dataset, val_dataset, _ = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])

        k_fold_cv_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset])
        val_size = int(0.2 * len(k_fold_cv_dataset))
        low_val_idx = fold_num * val_size
        high_val_idx = low_val_idx + val_size

        val_idxs = [i for i in range(low_val_idx, high_val_idx)]
        dummy_arr = np.ones(len(k_fold_cv_dataset))
        dummy_arr[val_idxs] = 0
        train_idxs = np.argwhere(dummy_arr != 0).squeeze()

        train_dataset = torch.utils.data.Subset(k_fold_cv_dataset, train_idxs)
        val_dataset = torch.utils.data.Subset(k_fold_cv_dataset, val_idxs)

        train_size = len(train_dataset)
        val_size = len(val_dataset)

        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, train_size, val_size, num_voxels


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

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    train_img_paths = img_paths[train_idxs]
    test_img_paths = img_paths[test_idxs]

    return train_dataloader, test_dataloader, train_size, test_size, num_voxels, train_img_paths, test_img_paths


# Function to create a list of length num_voxels, where each entry is a list containing the ROI(s) (may be none, 1, or multiple) that the voxel is in
# Should pass dataloader containing all voxels as argument
def get_voxel_roi_mappings(data_dir, subj_num, hemisphere, num_voxels):

    roi_masks_path = os.path.join(
        data_dir, "roi_masks", "Subj" + str(subj_num))

    # Get roi masks
    lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps = get_roi_mapping_files(
        roi_masks_path)
    if hemisphere == 'left':
        challenge_rois = lh_challenge_rois
    elif hemisphere == 'right':
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
                if roi_name_maps[roi_group][roi] == roi_name:
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


# Function to return n random ROIs (subject, hemisphere, and ROI), for use in hyperparameter search, saves a txt file listing them (make sure ROIs selected have >20 voxels)
def get_n_random_rois(project_dir, n):

    os.chdir(project_dir)

    subjs = list(range(1, 9))
    all_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1",
                "FFA-2", "mTL-faces", "aTL-faces", "OPA",
                "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
                "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    hemispheres = ['l', 'r']
    # Keep track of used roi keys(format="x_h_roi", where x is subj idx, h is hemisphere (left or right), and roi is roi name)
    used_roi_keys = set()
    for i in range(n):
        roi = all_rois[random.randint(0, len(all_rois)-1)]
        subj = random.randint(1, 8)
        hemisphere = 'left' if random.randint(0, 1) == 0 else 'right'
        # Get number of voxels in this roi
        _, _, _, _, _, _, num_voxels = get_dataloaders(
            project_dir, 'cpu', subj, hemisphere, roi, batch_size=1024)
        roi_key = str(subj) + "_" + hemisphere + "_" + roi
        while roi_key in used_roi_keys and num_voxels > 20:
            roi = all_rois[random.randint(0, len(all_rois)-1)]
            subj = random.randint(1, 8)
            hemisphere = 'left' if random.randint(0, 1) == 0 else 'right'
            # Get number of voxels in this roi
            _, _, _, _, _, _, num_voxels = get_dataloaders(
                project_dir, 'cpu', subj, hemisphere, roi, batch_size=1024)
            roi_key = str(subj) + "_" + hemisphere + "_" + roi
        used_roi_keys.add(roi_key)

    # Write rois to txt file
    f = open('random_rois.txt', 'w+')
    for random_roi_keys in used_roi_keys:
        f.write(random_roi_keys + '\n')
    f.close()


# Function to return all fmri data as np array given a dataloader
def get_fmri_from_dataloader(dataloader, num_images, num_voxels):
    fmri = np.zeros((num_images, num_voxels))
    for batch_index, data in enumerate(dataloader):
        fmri_batch = data[0]
        idxs = data[2]
        batch_size = fmri_batch.shape[0]
        if (batch_index == 0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        fmri[low_idx:high_idx] = fmri_batch.cpu().numpy()
        del fmri_batch, idxs
    return fmri


# Fit PCA to training images
def fit_pca(feature_extractor, dataloader, num_images, alex_out_layer, alex_out_size, is_cl_feature_extractor=False,
            is_reg_feature_extractor=False, make_dummy_fmri_data=False, num_voxels=0, pooled=False, subj_num=1):

    print("Fitting PCA...")
    features = np.zeros((num_images, alex_out_size))
    for batch_index, data in enumerate(dataloader):
        batch_size = data[0].shape[0]
        # For cross subject applications need to make dummy fmri data with correct number of voxels
        # that forward function of model subject is expecting
        if make_dummy_fmri_data:
            fmri_dummy = torch.zeros((batch_size, num_voxels))
        if batch_index == 0:
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # If a CL feature extractor, need to pass fmri data to forward function
        if is_cl_feature_extractor:
            with torch.no_grad():
                if make_dummy_fmri_data:
                    if pooled:
                        _, alex_out_dict = feature_extractor(
                            fmri_dummy, data[1], subj_num)
                    else:
                        _, alex_out_dict = feature_extractor(
                            fmri_dummy, data[1])
                else:
                    if pooled:
                        _, alex_out_dict = feature_extractor(
                            data[0], data[1], subj_num)
                    else:
                        _, alex_out_dict = feature_extractor(data[0], data[1])
            ft = alex_out_dict[alex_out_layer].detach()
            ft = torch.flatten(ft, start_dim=1)
        elif is_reg_feature_extractor:
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[1])
            ft = alex_out_dict[alex_out_layer].detach()
            ft = torch.flatten(ft, start_dim=1)
        else:
            with torch.no_grad():
                ft = feature_extractor(data[1])
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1)
                              for l in ft.values()]).cpu().detach().numpy()

        features[low_idx:high_idx] = ft
        del ft

    pca = PCA(n_components=1000, random_state=0).fit(features)
    return pca


# Given already fit PCA, extract PCA features
def extract_pca_features(feature_extractor, dataloader, pca, alex_out_layer, num_images, is_cl_feature_extractor=False,
                         is_reg_feature_extractor=False, make_dummy_fmri_data=False, num_voxels=0, pooled=False, subj_num=1):

    print("Extracting PCA Features...")
    features = np.zeros((num_images, 1000))
    for batch_index, data in enumerate(dataloader):
        batch_size = data[0].shape[0]
        print(data[2])
        if (batch_index == 0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # For cross subject applications need to make dummy fmri data with correct number of voxels
        # that forward function of model subject is expecting
        if make_dummy_fmri_data:
            fmri_dummy = torch.zeros((batch_size, num_voxels))
        # Extract features
        if is_cl_feature_extractor:
            with torch.no_grad():
                if make_dummy_fmri_data:
                    if pooled:
                        _, alex_out_dict = feature_extractor(
                            fmri_dummy, data[1], subj_num)
                    else:
                        _, alex_out_dict = feature_extractor(
                            fmri_dummy, data[1])
                else:
                    if pooled:
                        _, alex_out_dict = feature_extractor(
                            data[0], data[1], subj_num)
                    else:
                        _, alex_out_dict = feature_extractor(data[0], data[1])
            ft = alex_out_dict[alex_out_layer].detach()
            ft = torch.flatten(ft, start_dim=1)
        elif is_reg_feature_extractor:
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[1])
            ft = alex_out_dict[alex_out_layer].detach()
            ft = torch.flatten(ft, start_dim=1)
        else:
            with torch.no_grad():
                ft = feature_extractor(data[1])
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1)
                              for l in ft.values()]).cpu().detach().numpy()
        # Apply PCA transform
        ft = pca.transform(ft)
        features[low_idx:high_idx] = ft
        del ft
    return features


# Function to create file with number of voxels for each ROI (for a given hemisphere) across all subjects with that ROI present
# (which is used for pooled model with h_dim = average)
def make_voxels_counts_file(project_dir, hemisphere):

    device = 'cpu'
    rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
            "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA",
            "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
    avg_voxels_dict = {}

    for roi in rois:
        # Get dataloaders
        train_dataloaders_subjs = []
        num_voxels_subjs = []
        for subj_num in range(1, 9):
            train_dataloader, _, _, _, num_voxels = get_dataloaders(
                project_dir, device, subj_num, hemisphere, roi, batch_size=1024)
            if num_voxels != 0:
                train_dataloaders_subjs.append(train_dataloader)
                num_voxels_subjs.append(num_voxels)
        print(hemisphere, roi)
        print("Number of subjects with ROI present: " +
              str(len(num_voxels_subjs)))
        avg_voxels_dict[roi] = num_voxels_subjs
        print(num_voxels_subjs)

    hemisphere_abbr = 'l' if hemisphere == 'left' else 'r'
    results_file = os.path.join(
        project_dir, hemisphere_abbr + "h_voxel_counts_rois.joblib")
    joblib.dump(avg_voxels_dict, results_file)
