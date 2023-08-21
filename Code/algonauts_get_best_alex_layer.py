# Imports

import os
import sys

import joblib

import torch
from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

# Local imports
from algonauts_utils import get_dataloaders, get_dataloaders_unshuffled, get_roi_mapping_files, fit_pca


# Command line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to main folder of project", required=True)
parser.add_argument("--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument("--subj", help="Subject number, 1 through 8", type=int, required=True)
parser.add_argument("--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--batch-size", help="Batch size for training", type=int, default=1024)
parser.add_argument("--output", help="True to print results, False to not", type=bool, required=False, default=False)




# Given already fit PCA, extract and compress image features
def extract_features_and_fmri(feature_extractor, dataloader, pca, num_images, num_voxels):

    features = []
    print("Extracting PCA Features...")
    features = np.zeros((num_images, 1000))
    fmri = np.zeros((num_images, num_voxels))
    for batch_index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        fmri_batch = data[0]
        ft = feature_extractor(data[1])
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features[low_idx:high_idx] = ft
        fmri[low_idx:high_idx] = fmri_batch.cpu().numpy()
        del ft, fmri_batch
    return features, fmri



if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Load arguments
    project_dir = args.root
    device = args.device
    device = torch.device(device)
    subj_num = args.subj
    batch_size = args.batch_size
    hemisphere = args.hemisphere
    if (hemisphere != 'left' and hemisphere != 'right'):
        print("Invalid hemisphere, must be left or right")
        sys.exit()
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
    print_results = args.output
        
    all_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1",
            "FFA-2", "mTL-faces", "aTL-faces", "OPA",
              "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
           "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal", "All vertices"]
    
    roi_map_dir = project_dir + r"/roi_masks/Subj" + str(subj_num)
    save_dir = project_dir + r"/best_alex_out_layers/Subj" + str(subj_num) 
    os.chdir(save_dir)
    
    best_alex_layers_dict = {}
    alex_out_layers = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"]
    alex_out_layer_dims = {"features.2":46656, "features.5":32448, "features.7":64896, "features.9":43264, "features.12":9216, "classifier.2":4096, "classifier.5":4096, "classifier.6":1000}

    layer_dicts = []
    
    # Get dataloaders for all voxels
    
    train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders_unshuffled(project_dir, device, subj_num, hemisphere, "all", batch_size=1024)
    
    for layer in alex_out_layers:

        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
        model.to(device) 
        model.eval()
        feature_extractor = create_feature_extractor(model, return_nodes=[layer])

        # Fit and extract image features from PCA
        pca = fit_pca(feature_extractor, train_dataloader, train_size, alex_out_layer_dims[layer], batch_size=1024)
        features_train, fmri_train = extract_features_and_fmri(feature_extractor, train_dataloader, pca, train_size, num_voxels)
        features_test, fmri_test = extract_features_and_fmri(feature_extractor, test_dataloader, pca, test_size, num_voxels)
        
        # Fit linear regressions on the training data
        lin_reg = LinearRegression().fit(features_train, fmri_train)
        # Use fitted linear regressions to predict the validation and test fmri data
        fmri_test_pred = lin_reg.predict(features_test)

        # Compute encoding accuracy (correlation)
        # Empty correlation array of shape: (num_voxels)
        correlations = np.zeros(num_voxels)
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(num_voxels)):
            correlations[v] = corr(fmri_test_pred[:,v], fmri_test[:,v])[0]


        # Get roi masks to break results down by ROIs
        lh_challenge_rois, rh_challenge_rois, roi_names, roi_name_maps = get_roi_mapping_files(roi_map_dir)
        if (hemisphere=='left'):
            challenge_rois = lh_challenge_rois
        else:
            challenge_rois = rh_challenge_rois
        
        
        # Select the correlation results vertices of each ROI
        roi_names = []
        roi_correlations = []
        for r1 in range(len(challenge_rois)):
            for r2 in roi_name_maps[r1].items():
                if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    roi_idx = np.where(challenge_rois[r1] == r2[0])[0]
                    roi_correlations.append(correlations[roi_idx])
        roi_names.append('All vertices')
        roi_correlations.append(correlations)
        # Compute median correlations
        median_roi_correlation = [np.median(roi_correlations[r])
          for r in range(len(roi_correlations))]

        # Add median correlations to dictionary
        correlations_dict = {}
        print(layer)
        for roi_idx in range(len(roi_names)):
            correlations_dict[roi_names[roi_idx]] = str(median_roi_correlation[roi_idx])
            if (print_results):
                print(roi_names[roi_idx] + ": " + str(median_roi_correlation[roi_idx]))
        layer_dicts.append(correlations_dict)
        
        
    # Save the best layer (highest correlation) for each ROI
    best_layer_corrs_dict = {}
    best_layer_dict = {}
    for i in range(len(all_rois)):
        best_layer_corrs_dict[all_rois[i]] = 0
        best_layer_dict[all_rois[i]] = "None"

    for layer_idx in range(len(layer_dicts)):
        layer_dict = layer_dicts[layer_idx]
        layer = alex_out_layers[layer_idx]
        for roi in all_rois:
            if (float(layer_dict[roi]) > best_layer_corrs_dict[roi]):
                best_layer_corrs_dict[roi] = float(layer_dict[roi])
                best_layer_dict[roi] = layer
                
    # Save best layers
    save_file_name = "subj" + str(subj_num) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
    joblib.dump(best_layer_dict, save_file_name)
