# Imports

import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
#from torchsummary import summary
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
import joblib
import time
import itertools

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

import torchvision.models as models
#import torch.nn.functional as F

from torchmetrics.functional import pairwise_cosine_similarity
from scipy.stats import pearsonr as corr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

#from IPython.utils import io

#import matplotlib.pyplot as plt

from tqdm import tqdm

import torchextractor as tx
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

from sklearn.manifold import TSNE
#from matplotlib.pyplot import imshow

#import pandas as pd
#import seaborn as sns


# Local imports
from algonauts_utils import get_dataloaders, get_dataloaders_unshuffled, fit_pca, extract_pca_features, get_fmri_from_dataloader, get_dataloaders_with_img_paths
from algonauts_models import CLR_model



# Function to print results for control linear encoding, CL alex fine-tuned feature 
# encoding with regularization, and neural network regression alex fined-tuned feature
# encoding with regularization, all for subject-specific models
def get_results_single_subj(project_dir, device, subj_num, hemisphere, roi):
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
    
    # Get dataloaders
    train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders_unshuffled(project_dir, 
                                                                device, subj_num, hemisphere, roi, 1024)
    if (num_voxels==0):
        print("Empty ROI")
        return -1,-1,-1,-1,-1,-1,-1
    elif (num_voxels<20):
        print("Too few voxels")
        return -1,-1,-1,-1,-1,-1,-1
    
    # Get training and testing fmri and numpy arrays
    train_fmri = get_fmri_from_dataloader(train_dataloader, train_size, num_voxels)
    test_fmri = get_fmri_from_dataloader(test_dataloader, test_size, num_voxels)
    
    
    # Get control linear encoding model predictions:
    # Load best alexnet layer for control linear encoding model, create feature extractor
    print("Getting control linear encoding model predictions...")
    best_alex_out_layer_path = project_dir + r"/best_alex_out_layers/Subj" + str(subj_num) + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
    best_alex_out_layer_dict = joblib.load(best_alex_out_layer_path)
    alex_out_layer_dims = {"features.2":46656, "features.5":32448, "features.7":64896, "features.9":43264,
                           "features.12":9216, "classifier.2":4096, "classifier.5":4096, "classifier.6":1000}
    best_alex_layer = best_alex_out_layer_dict[roi]
    alex_out_size = alex_out_layer_dims[best_alex_layer]
    
    alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
    alex.to(device) # send the model to the chosen device 
    alex.eval() # set the model to evaluation mode
    feature_extractor = create_feature_extractor(alex, return_nodes=[best_alex_layer])
    del alex
    
    #Fit PCA using feature extractor
    pca = fit_pca(feature_extractor, train_dataloader, train_size, alex_out_size, batch_size=1024)
    
    # Get training and testing image pca features
    train_pca_features = extract_pca_features(feature_extractor, pca, train_size, train_dataloader, 
                                                    is_img_dataloader=False, batch_size=1024)
    test_pca_features = extract_pca_features(feature_extractor, pca, test_size, test_dataloader, 
                                                    is_img_dataloader=False, batch_size=1024)
    
    # Fit control linear encoding model, get test predictions
    #control_linear_model = LinearRegression().fit(train_pca_features, train_fmri)
    control_linear_model = Ridge(alpha=1000).fit(train_pca_features, train_fmri)
    
    ctrl_preds = control_linear_model.predict(test_pca_features)

    del feature_extractor, train_pca_features, test_pca_features, control_linear_model
    
    
    
    # Get preds using tuned alexnet from CL:
    # Load CL model
    print("Getting CL predictions...")
    cl_model_dir = project_dir + r"/cl_models/Subj" + str(subj_num)
    cl_model_path = cl_model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_model_e30.pt" 
    cl_model = torch.load(cl_model_path).to(device)
    
    # Create feature extractor
    feature_extractor = tx.Extractor(cl_model, ["alex.classifier.6"]).to(device)
    del cl_model
    
    # Get training and testing tuned alexnet features
    train_features = np.zeros((train_size, 1000))
    for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            _, alex_out_dict = feature_extractor(data[0], data[1])
        ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
        train_features[low_idx:high_idx] = ft
        del ft

    test_features = np.zeros((test_size, 1000))
    for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            _, alex_out_dict = feature_extractor(data[0], data[1])
        ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
        test_features[low_idx:high_idx] = ft
        del ft

    cl_encoding_model = Ridge(alpha=10000).fit(train_features, train_fmri)
    cl_preds = cl_encoding_model.predict(test_features)
    
    del feature_extractor, train_features, test_features, cl_encoding_model
    
    
    
    # Get predictions using tuned alexnet from regression model:
    # Load regression model
    print("Getting regression predictions...")
    reg_model_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(subj_num)
    reg_model_path = reg_model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e75.pt" 
    reg_model = torch.load(reg_model_path).to(device)
    
    # Create feature extractor
    feature_extractor = tx.Extractor(reg_model, ["alex.classifier.6"]).to(device)
    del reg_model
    
    # Get training and testing tuned alexnet features
    train_features = np.zeros((train_size, 1000))
    for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            _, alex_out_dict = feature_extractor(data[1])
        ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
        train_features[low_idx:high_idx] = ft
        del ft
    test_features = np.zeros((test_size, 1000))
    for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            _, alex_out_dict = feature_extractor(data[1])
        ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
        test_features[low_idx:high_idx] = ft
        del ft
        
    reg_encoding_model = Ridge(alpha=10000).fit(train_features, train_fmri)
    reg_preds = reg_encoding_model.predict(test_features)
    
    del feature_extractor, train_features, test_features, reg_encoding_model
    
    
    
    # Compute mean correlations for all methods
    ctrl_corrs = np.zeros(num_voxels)
    cl_corrs = np.zeros(num_voxels)
    reg_corrs = np.zeros(num_voxels)
    print("Computing correlations...")
    for v in tqdm(range(num_voxels)):
        ctrl_corrs[v] = corr(test_fmri[:, v], ctrl_preds[:, v])[0]
        cl_corrs[v] = corr(test_fmri[:, v], cl_preds[:, v])[0]
        reg_corrs[v] = corr(test_fmri[:, v], reg_preds[:, v])[0]

    ctrl_avg = ctrl_corrs.mean()
    cl_avg = cl_corrs.mean()
    reg_avg = reg_corrs.mean()
    
    # Get percentage of improved voxels
    cl_ctrl_improved_percentage = np.count_nonzero(cl_corrs - ctrl_corrs > 0) / num_voxels
    cl_reg_improved_percentage = np.count_nonzero(cl_corrs - reg_corrs > 0) / num_voxels
    
    cl_vs_ctrl_voxel_differences = cl_corrs - ctrl_corrs
    cl_vs_reg_voxel_differences = cl_corrs - reg_corrs
    
    print(roi + ":")
    print("Average control correlation: " + str(np.round(ctrl_avg,3)))
    print("Average regression correlation: " + str(np.round(reg_avg,3)))
    print("Average CL correlation: " + str(np.round(cl_avg,3)))

    
    print("% of voxels improved versus control: " + str(np.round(cl_ctrl_improved_percentage,3)))
    print("% of voxels improved versus regression: " + str(np.round(cl_reg_improved_percentage,3)))
    
    del train_fmri, test_fmri, train_dataloader, test_dataloader
    return ctrl_avg, reg_avg, cl_avg, cl_ctrl_improved_percentage, cl_reg_improved_percentage, cl_vs_ctrl_voxel_differences, cl_vs_reg_voxel_differences



def get_results_single_subj_all_rois(project_dir, device, subj_num, hemisphere, save=True, exclude_rois=None):
    
    all_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
              "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA",
         "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
       "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'

    if (save):
        ctrl_results = {}
        reg_results = {}
        cl_results = {}
        improved_percentages_vs_ctrl = {}
        improved_percentages_vs_reg = {}
        voxel_differences_vs_ctrl = {}
        voxel_differences_vs_reg = {}
          
    for roi in all_rois:
        ctrl_avg, reg_avg, cl_avg, cl_ctrl_improved_percentage, cl_reg_improved_percentage, cl_vs_ctrl_voxel_differences, cl_vs_reg_voxel_differences = get_results_single_subj(project_dir, device, subj_num, hemisphere, roi)
        if (save):
            ctrl_results[roi] = ctrl_avg
            reg_results[roi] = reg_avg
            cl_results[roi] = cl_avg
            improved_percentages_vs_ctrl[roi] = cl_ctrl_improved_percentage
            improved_percentages_vs_reg[roi] = cl_reg_improved_percentage
            voxel_differences_vs_ctrl[roi] = cl_vs_ctrl_voxel_differences
            voxel_differences_vs_reg[roi] = cl_vs_reg_voxel_differences

    # Save results
    if (save):
        
        save_folder = project_dir + "/results/Subj" + str(subj_num) 
        
        ctrl_results_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_ctrl_results.joblib"
        cl_results_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_cl_results.joblib"
        reg_results_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_reg_results.joblib"
        ctrl_improved_percentages_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_cl_vs_ctrl_improved_percentages.joblib"
        reg_improved_percentages_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_cl_vs_reg_improved_percentages.joblib"
        ctrl_improved_voxels_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_cl_vs_ctrl_voxel_differences.joblib"
        reg_improved_voxels_save_file = save_folder + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_cl_vs_reg_voxel_differences.joblib"
        
        joblib.dump(ctrl_results, ctrl_results_save_file)
        joblib.dump(cl_results, cl_results_save_file)
        joblib.dump(reg_results, reg_results_save_file)
        joblib.dump(improved_percentages_vs_ctrl, ctrl_improved_percentages_save_file)
        joblib.dump(improved_percentages_vs_reg, reg_improved_percentages_save_file)
        joblib.dump(voxel_differences_vs_ctrl, ctrl_improved_voxels_save_file)
        joblib.dump(voxel_differences_vs_reg, reg_improved_voxels_save_file)
        
        
        

# Function to get results for using models cross-subject
def get_results_cross_subj(project_dir, device, hemisphere, roi, save=True, num_subjs=8, reg=False, training_results=False):
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
    
    results_mat = np.zeros((num_subjs, num_subjs))
    voxel_improvement_percentages_ctrl_mat = np.zeros((num_subjs, num_subjs))
    #voxel_improvement_percentages_reg_mat = np.zeros((num_subjs, num_subjs))
    
    
    # Get control predictions for this ROI for each subject
    ctrl_preds_all_subjs = []
    #reg_preds_all_subjs = []
    for subj_idx in range(num_subjs):
        
        subj_num = subj_idx + 1
 
        print("Getting control linear encoding model predictions...")
        best_alex_out_layer_path = project_dir + r"/best_alex_out_layers/Subj" + str(subj_num) + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
        best_alex_out_layer_dict = joblib.load(best_alex_out_layer_path)
        alex_out_layer_dims = {"features.2":46656, "features.5":32448, "features.7":64896, "features.9":43264,
                               "features.12":9216, "classifier.2":4096, "classifier.5":4096, "classifier.6":1000}
        best_alex_layer = best_alex_out_layer_dict[roi]
        alex_out_size = alex_out_layer_dims[best_alex_layer]

        alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
        alex.to(device) 
        alex.eval() 
        feature_extractor = create_feature_extractor(alex, return_nodes=[best_alex_layer])
        del alex
        
        train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders_unshuffled(project_dir, device, subj_num, hemisphere, roi, 1024)
        
        #Fit PCA using feature extractor
        pca = fit_pca(feature_extractor, train_dataloader, train_size, alex_out_size, batch_size=1024)

        # Get training and testing image pca features
        train_pca_features = extract_pca_features(feature_extractor, pca, train_size, train_dataloader, 
                                                        is_img_dataloader=False, batch_size=1024)
        test_pca_features = extract_pca_features(feature_extractor, pca, test_size, test_dataloader, 
                                                        is_img_dataloader=False, batch_size=1024)
        
        print("Loading fmri data...")
        train_fmri = get_fmri_from_dataloader(train_dataloader, train_size, num_voxels)
        test_fmri = get_fmri_from_dataloader(test_dataloader, test_size, num_voxels)


        # Fit control linear encoding model, get test predictions
        control_linear_model = Ridge(alpha=1000).fit(train_pca_features, train_fmri)

        if (training_results):
            ctrl_preds = control_linear_model.predict(train_pca_features)
        else:
            ctrl_preds = control_linear_model.predict(test_pca_features)
        
        ctrl_preds_all_subjs.append(ctrl_preds)
        
        
        """
        # Load regression model
        print("Getting regression predictions...")
        reg_model_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(subj_num)
        reg_model_path = reg_model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e75.pt" 
        reg_model = torch.load(reg_model_path).to(device)

        # Create feature extractor
        feature_extractor = tx.Extractor(reg_model, ["alex.classifier.6"]).to(device)
        del reg_model

        # Get training and testing tuned alexnet features
        train_features = np.zeros((train_size, 1000))
        for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_size = data[0].shape[0]
            if (batch_index==0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
            # Extract features
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[1])
            ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
            train_features[low_idx:high_idx] = ft
            del ft
        test_features = np.zeros((test_size, 1000))
        for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            batch_size = data[0].shape[0]
            if (batch_index==0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
            # Extract features
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[1])
            ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
            test_features[low_idx:high_idx] = ft
            del ft

        reg_encoding_model = Ridge(alpha=10000).fit(train_features, train_fmri)
        reg_preds = reg_encoding_model.predict(test_features)
        
        reg_preds_all_subjs.append(reg_preds)
        
        

        

        del feature_extractor, train_pca_features, test_pca_features, control_linear_model, train_fmri, test_fmri, reg_encoding_model
        """
    
        
        
    
    for model_subj_idx in range(num_subjs):
        
        model_subj = model_subj_idx + 1
        
        # Load model for model subject
        if (not reg):
            model_dir = project_dir + r"/cl_models/Subj" + str(model_subj)
            model_path = model_dir + r"/subj" + str(model_subj) + "_" + hemisphere_abbr + "h_" + roi + "_model_e30.pt" 
            model = torch.load(model_path).to(device)
        else:
            model_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(model_subj)
            model_path = model_dir + r"/subj" + str(model_subj) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e75.pt" 
            model = torch.load(model_path).to(device)
        
        # Create feature extractor
        feature_extractor = tx.Extractor(model, ["alex.classifier.6"]).to(device)
        del model
        
        train_dataloader, test_dataloader, train_size, test_size, num_voxels_model_subj = get_dataloaders_unshuffled(project_dir, device, model_subj, hemisphere, roi, 1024)
        

        for target_subj_idx in range(num_subjs):
            
            target_subj = target_subj_idx + 1
            
            # Get dataloaders for target subj
            train_dataloader, test_dataloader, train_size, test_size, num_voxels_target_subj = get_dataloaders_unshuffled(project_dir, device, target_subj, hemisphere, roi, 1024)
            
            if (num_voxels_target_subj==0):
                results_mat[model_subj_idx, target_subj_idx] = -1
            else:
                # Get training and testing fmri for target subj
                print("Loading fmri data...")
                train_fmri = get_fmri_from_dataloader(train_dataloader, train_size, num_voxels_target_subj)
                test_fmri = get_fmri_from_dataloader(test_dataloader, test_size, num_voxels_target_subj)

                # Get training and testing tuned alexnet features
                print("Getting train image features...")
                train_features = np.zeros((train_size, 1000))
                for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                    batch_size = data[0].shape[0]
                    if (batch_index==0):
                        low_idx = 0
                        high_idx = batch_size
                    else:
                        low_idx = high_idx
                        high_idx += batch_size
                    # Extract features
                    with torch.no_grad():
                        if (not reg):
                            fmri_dummy = torch.zeros((batch_size, num_voxels_model_subj)).to(device)
                            _, alex_out_dict = feature_extractor(fmri_dummy, data[1])
                        else:
                            _, alex_out_dict = feature_extractor(data[1])
                    ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                    train_features[low_idx:high_idx] = ft
                    del ft

                print("Getting test image features...")
                test_features = np.zeros((test_size, 1000))
                for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    batch_size = data[0].shape[0]
                    if (batch_index==0):
                        low_idx = 0
                        high_idx = batch_size
                    else:
                        low_idx = high_idx
                        high_idx += batch_size
                    # Extract features
                    with torch.no_grad():
                        if (not reg):
                            fmri_dummy = torch.zeros((batch_size, num_voxels_model_subj)).to(device)
                            _, alex_out_dict = feature_extractor(fmri_dummy, data[1])
                        else:
                            _ , alex_out_dict = feature_extractor(data[1])
                    ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                    test_features[low_idx:high_idx] = ft
                    del ft

                # Fit encoding model and use it to predict test fmri
                print("Fitting linear encoding model...")
                encoding_model = Ridge(alpha=10000).fit(train_features, train_fmri)
                
                if (training_results):
                    cs_preds = encoding_model.predict(train_features)
                else:
                    cs_preds = encoding_model.predict(test_features)

                # Evaluate mean correlation accuracy
                ctrl_corrs = np.zeros(num_voxels_target_subj)
                cs_corrs = np.zeros(num_voxels_target_subj)
                
                ctrl_preds = ctrl_preds_all_subjs[target_subj_idx]
                print("Computing correlations...")
                for v in tqdm(range(num_voxels_target_subj)):
                    if (training_results):
                        ctrl_corrs[v] = corr(train_fmri[:, v], ctrl_preds[:, v])[0]
                        cs_corrs[v] = corr(train_fmri[:, v], cs_preds[:, v])[0]
                    else:
                        ctrl_corrs[v] = corr(test_fmri[:, v], ctrl_preds[:, v])[0]
                        cs_corrs[v] = corr(test_fmri[:, v], cs_preds[:, v])[0]

                # Save mean correlation
                mean_ctrl_corr = ctrl_corrs.mean()
                mean_cs_corr = cs_corrs.mean()
                
                print(mean_ctrl_corr, mean_cs_corr)
                
                results_mat[model_subj_idx, target_subj_idx] = mean_cs_corr
                print(results_mat)

                
                voxel_improvement_percentages_ctrl_mat[model_subj_idx, target_subj_idx] = np.count_nonzero(cs_corrs - ctrl_corrs > 0) / num_voxels_target_subj
                
                
                print(voxel_improvement_percentages_ctrl_mat)
      
            
    if (save):
        if (not reg):
            if (training_results):
                accuracies_save_path = project_dir + "/results/" + roi + "_" + hemisphere_abbr + "h_cross_subject_cl_results_mat_training.npy"
                voxel_improvement_percentages_ctrl_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cs_voxel_improvement_percentages_cl_mat_training.npy"
            else:
                accuracies_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cross_subject_cl_results_mat.npy"
                voxel_improvement_percentages_ctrl_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cs_voxel_improvement_percentages_cl_mat.npy"
        else:
            if (training_results):
                accuracies_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cross_subject_reg_results_mat_training.npy"
                voxel_improvement_percentages_ctrl_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cs_voxel_improvement_percentages_reg_mat_training.npy" 
            else:
                accuracies_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cross_subject_reg_results_mat.npy"
                voxel_improvement_percentages_ctrl_save_path = project_dir + "/results/" + roi + hemisphere_abbr + "h_cs_voxel_improvement_percentages_reg_mat.npy"
       
        np.save(accuracies_save_path, results_mat)
        np.save(voxel_improvement_percentages_ctrl_save_path, voxel_improvement_percentages_ctrl_mat)




# Function to visualize similar images from CL-tuned AlexNet using t-SNE
def tsne_visualization(subj, hemisphere, roi, control_alex=False):
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
        
    # Get dataloaders 
    train_dataloader, test_dataloader, train_size, test_size, num_voxels, train_img_paths, test_img_paths = get_dataloaders_with_img_paths(project_dir, device, subj, hemisphere, roi, batch_size=1024)

    if (num_voxels==0):
        print("Empty ROI")
        return
    
    if (control_alex):
        best_alex_out_layer_path = project_dir + r"/best_alex_out_layers/Subj" + str(subj) + r"/subj" + str(subj) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
        best_alex_out_layer_dict = joblib.load(best_alex_out_layer_path)
        alex_out_layer_dims = {"features.2":46656, "features.5":32448, "features.7":64896, "features.9":43264,
                               "features.12":9216, "classifier.2":4096, "classifier.5":4096, "classifier.6":1000}
        #best_alex_layer = best_alex_out_layer_dict[roi]
        best_alex_layer = 'classifier.6'
        alex_out_size = alex_out_layer_dims[best_alex_layer]

        alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
        alex.to(device) # send the model to the chosen device 
        alex.eval() # set the model to evaluation mode
        feature_extractor = create_feature_extractor(alex, return_nodes=[best_alex_layer])
        del alex

        #Fit pca to training images using feature extractor
        pca = fit_pca(feature_extractor, train_dataloader, train_size, alex_out_size, batch_size=1024)

        # Get pca features for test images
        test_features = extract_pca_features(feature_extractor, pca, test_size, test_dataloader, 
                                                        is_img_dataloader=False, batch_size=1024)
        del feature_extractor
        
    else:
        # Load CL model
        print("Extracting image features from tuned AlexNet...")
        cl_model_dir = project_dir + r"/cl_models/Subj" + str(subj)
        cl_model_path = cl_model_dir + r"/subj" + str(subj) + "_" + hemisphere_abbr + "h_" + roi + "_model_e30.pt" 
        cl_model = torch.load(cl_model_path).to(device)

        # Create feature extractor
        feature_extractor = tx.Extractor(cl_model, ["alex.classifier.6"]).to(device)
        del cl_model

        # Get testing images, image features from tuned alexnet
        test_features = np.zeros((test_size, 1000))
        for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            batch_size = data[0].shape[0]
            if (batch_index==0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
            # Extract features
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[0], data[1])
            ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
            test_features[low_idx:high_idx] = ft
            del ft
            
        del feature_extractor
        
        
    
    # Fit tsne, scale tsne components
    tsne = TSNE(n_components=2).fit_transform(test_features)
    
    x = tsne[:, 0]
    y = tsne[:, 1]
    
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    y = (y-np.min(y)) / (np.max(y) - np.min(y))
    
    width = 4000
    height = 4000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for img_path, img_x, img_y in zip(test_img_paths, x, y):
        tile = Image.open(img_path)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*img_x), int((height-max_dim)*img_y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (8,8))
    imshow(full_image)
    
    
    

# Function to get violin plots comparing distribution of voxel encoding accuracies for given subj ROIs
def get_violin_plot(subj_num, hemisphere, rois, reg_control=False):
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
    
    # Dictionaries to store results (as a dataframe to be used for plotting) for each roi
    dfs_dict = {}
    
    for roi in rois:
        # Get dataloaders
        train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders(project_dir, 
                                                                    device, subj_num, hemisphere, roi, 1024)
        if (num_voxels==0):
            print("Empty ROI")
            return -1,-1,-1
        
        # Get training and testing fmri and numpy arrays
        train_fmri = get_fmri_from_dataloader(train_dataloader, train_size, num_voxels)
        test_fmri = get_fmri_from_dataloader(test_dataloader, test_size, num_voxels)


        if (reg_control):
            # Get predictions using tuned alexnet from regression model:
            # Load regression model
            print("Getting regression predictions...")
            reg_model_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(subj_num)
            reg_model_path = reg_model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e75.pt" 
            reg_model = torch.load(reg_model_path).to(device)

            # Create feature extractor
            feature_extractor = tx.Extractor(reg_model, ["alex.classifier.6"]).to(device)
            del reg_model

            # Get training and testing tuned alexnet features
            train_features = np.zeros((train_size, 1000))
            for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                train_features[low_idx:high_idx] = ft
                del ft
            test_features = np.zeros((test_size, 1000))
            for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                test_features[low_idx:high_idx] = ft
                del ft

            reg_encoding_model = Ridge(alpha=10000).fit(train_features, train_fmri)
            reg_preds = reg_encoding_model.predict(test_features)

            del feature_extractor, train_features, test_features, reg_encoding_model


            
        else:
            # Get control linear encoding model predictions:
            # Load best alexnet layer for control linear encoding model, create feature extractor
            print("Getting control linear encoding model predictions...")
            best_alex_out_layer_path = project_dir + r"/best_alex_out_layers/Subj" + str(subj_num) + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
            best_alex_out_layer_dict = joblib.load(best_alex_out_layer_path)
            alex_out_layer_dims = {"features.2":46656, "features.5":32448, "features.7":64896, "features.9":43264,
                                   "features.12":9216, "classifier.2":4096, "classifier.5":4096, "classifier.6":1000}
            best_alex_layer = best_alex_out_layer_dict[roi]
            alex_out_size = alex_out_layer_dims[best_alex_layer]

            alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
            alex.to(device) # send the model to the chosen device 
            alex.eval() # set the model to evaluation mode
            feature_extractor = create_feature_extractor(alex, return_nodes=[best_alex_layer])
            del alex

            #Fit PCA using feature extractor
            pca = fit_pca(feature_extractor, train_dataloader, train_size, alex_out_size, batch_size=1024)

            # Get training and testing image pca features
            train_pca_features = extract_pca_features(feature_extractor, pca, train_size, train_dataloader, 
                                                            is_img_dataloader=False, batch_size=1024)
            test_pca_features = extract_pca_features(feature_extractor, pca, test_size, test_dataloader, 
                                                            is_img_dataloader=False, batch_size=1024)

            # Fit control linear encoding model, get test predictions
            control_linear_model = LinearRegression().fit(train_pca_features, train_fmri)
            control_linear_encoding_preds = control_linear_model.predict(test_pca_features)

            del feature_extractor, train_pca_features, test_pca_features, control_linear_model
        


        # Get preds using tuned alexnet from CL:
        # Load CL model
        print("Getting CL predictions...")
        cl_model_dir = project_dir + r"/cl_models/Subj" + str(subj_num)
        cl_model_path = cl_model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_model_e30.pt" 
        cl_model = torch.load(cl_model_path).to(device)

        # Create feature extractor
        feature_extractor = tx.Extractor(cl_model, ["alex.classifier.6"]).to(device)
        del cl_model

        # Get training and testing tuned alexnet features
        train_features = np.zeros((train_size, 1000))
        for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_size = data[0].shape[0]
            if (batch_index==0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
            # Extract features
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[0], data[1])
            ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
            train_features[low_idx:high_idx] = ft
            del ft

        test_features = np.zeros((test_size, 1000))
        for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            batch_size = data[0].shape[0]
            if (batch_index==0):
                low_idx = 0
                high_idx = batch_size
            else:
                low_idx = high_idx
                high_idx += batch_size
            # Extract features
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[0], data[1])
            ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
            test_features[low_idx:high_idx] = ft
            del ft

        cl_encoding_model = Ridge(alpha=10000).fit(train_features, train_fmri)
        cl_encoding_preds = cl_encoding_model.predict(test_features)

        del feature_extractor, train_features, test_features, cl_encoding_model


        # Compute mean correlations for all methods
        ctrl_corrs = np.zeros(num_voxels)
        cl_corrs = np.zeros(num_voxels)
        #reg_corrs = np.zeros(num_voxels)
        print("Computing correlations...")
        for v in tqdm(range(num_voxels)):
            ctrl_corrs[v] = corr(test_fmri[:, v], control_linear_encoding_preds[:, v])[0]
            cl_corrs[v] = corr(test_fmri[:, v], cl_encoding_preds[:, v])[0]
            #reg_corrs[v] = corr(test_fmri[:, v], reg_encoding_preds[:, v])[0]

        
        # Create data frame
        all_corrs = np.zeros((num_voxels, 2))
        all_corrs[:, 0] = ctrl_corrs
        all_corrs[:, 1] = cl_corrs
        df = pd.DataFrame(data=all_corrs, columns=["Ctrl", "CL"])
        df = df.melt().assign(x=roi)
        df = df.rename(columns={'variable': 'Method', 'value':'Correlation', 'x':'ROI'}) 
        
        dfs_dict[roi] = df
        
        
    # Concatenate all roi dfs into 1 df
    dfs = []
    for roi in dfs_dict.keys():
        dfs.append(dfs_dict[roi])
        
    final_df = pd.concat(dfs)
    sns.violinplot(final_df, x='ROI', y='Correlation', hue='Method', split=True, inner=None)
    plt.title("Ctrl vs. CL Correlations")
    plt.show()
    
    return dfs_dict




def get_people_classifier_results(project_dir, device, subj_num, hemisphere, rois=None, all_rois=True):
    
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
    
    # Create dictionaries for results
    results_dict = {}
    
    if (all_rois):
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
              "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA",
         "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
       "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]

    
    for roi in rois:
        
        # Get dataloaders, untuned alexnet feature extractor
        train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders_unshuffled(project_dir, device, subj_num, hemisphere, roi, batch_size=1024) 
        
        if (num_voxels > 20):

            # Load people labels
            labels_path = project_dir + "/coco_data/subj" + str(subj_num) + "_people_labels.npy"
            labels = np.load(labels_path)

            train_idxs = torch.arange(0, train_size)
            test_idxs = torch.arange(train_size, train_size+test_size)

            train_labels = labels[train_idxs]
            test_labels = labels[test_idxs]


            # Load untuned alexnet model
            alex = models.alexnet(weights='DEFAULT')
            #alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=)
            alex.to(device) 
            alex.eval() 
            feature_extractor = create_feature_extractor(alex, return_nodes=['classifier.6']).to(device)
            del alex

            # Get untuned alexnet features
            train_features_untuned = np.zeros((train_size, 1000))
            for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    ft = feature_extractor(data[1])
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()
                train_features_untuned[low_idx:high_idx] = ft
                del ft

            test_features_untuned = np.zeros((test_size, 1000))
            for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    ft = feature_extractor(data[1])
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()
                test_features_untuned[low_idx:high_idx] = ft
                del ft
            del feature_extractor



            # Get accuracy for untuned alexnet
            svm_untuned = LinearSVC().fit(train_features_untuned, train_labels)
            untuned_acc = svm_untuned.score(test_features_untuned, test_labels)
            print("Untuned score: " + str(untuned_acc))
            results_dict['Untuned'] = untuned_acc



            # Get tuned alexnet scores from regression models
            reg_model_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(subj_num)
            reg_model_path = reg_model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e75.pt" 
            reg_model = torch.load(reg_model_path).to(device)

            # Create feature extractor
            feature_extractor = tx.Extractor(reg_model, ["alex.classifier.6"]).to(device)
            del reg_model

            # Get training and testing tuned alexnet features
            train_features_tuned = np.zeros((train_size, 1000))
            for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                train_features_tuned[low_idx:high_idx] = ft
                del ft
            test_features_tuned = np.zeros((test_size, 1000))
            for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                test_features_tuned[low_idx:high_idx] = ft
                del ft

            svm_tuned = LinearSVC().fit(train_features_tuned, train_labels)
            tuned_acc = svm_tuned.score(test_features_tuned, test_labels)
            print(roi + "-tuned score (reg): " + str(tuned_acc))

            key = roi + "_reg"
            results_dict[key] = tuned_acc


            # Get tuned alexnet features using CL model 
            cl_model_path = project_dir + r"/cl_models/Subj" + str(subj_num) + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_model_e30.pt" 
            cl_model = torch.load(cl_model_path).to(device)
            feature_extractor = tx.Extractor(cl_model, ["alex.classifier.6"]).to(device)
            del cl_model

            train_features_tuned = np.zeros((train_size, 1000))
            for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[0], data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                train_features_tuned[low_idx:high_idx] = ft
                del ft

            test_features_tuned = np.zeros((test_size, 1000))
            for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    _, alex_out_dict = feature_extractor(data[0], data[1])
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                test_features_tuned[low_idx:high_idx] = ft
                del ft


            svm_tuned = LinearSVC().fit(train_features_tuned, train_labels)
            tuned_acc = svm_tuned.score(test_features_tuned, test_labels)
            print(roi + "-tuned score (cl): " + str(tuned_acc))

            key = roi + "_cl"
            results_dict[key] = tuned_acc
        
        else:
            results_dict[key] = -1
                                    
    
    # Save results
    save_path = project_dir + "/results/Subj" + str(subj_num) + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_people_classifier_results.joblib"
    joblib.dump(results_dict, save_path)

    
    
# Get results for image classification task
def image_classification_results(project_dir, subj_num, hemisphere, rois, device, tuning_method='cl', dataset_name='caltech256', save=False):
    
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
    
    save_path = project_dir + "/results/Subj" + str(subj_num) + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + dataset_name + "_" + tuning_method + "_results.joblib"
    
    # Seed RNG, define image transforms for alexnet
    torch.manual_seed(0)
    alex_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
                ])
    
    # Initialize the Weight Transforms
    from torchvision.models import AlexNet_Weights
    
    # Load the Data
    if (dataset_name=='caltech256'):
        image_dir = project_dir + "/caltech256"
        dataset = torchvision.datasets.ImageFolder(root=image_dir, transform=alex_transform)
    elif (dataset_name=='places365'):
        image_dir = project_dir + "/places365"
        dataset = torchvision.datasets.Places365(root=image_dir, split='val', small=True, transform=alex_transform)
    elif (dataset_name=='sun397'):
        image_dir = project_dir + "/sun397"
        dataset = torchvision.datasets.SUN397(root=image_dir, transform=alex_transform, download=True)
    elif (dataset_name=='imagenet'):
        image_dir = project_dir + "/imagenet"
        #dataset = torchvision.datasets.ImageNet(root=image_dir, split='val', transform=alex_transform)
        dataset = torchvision.datasets.ImageNet(root=image_dir, split='val', transform=alex_transform)
        
    total_num_images = len(dataset)
    shuffled_idxs = torch.randperm(total_num_images)
    
    train_size = int(0.85 * total_num_images)
    test_size = total_num_images - train_size
    
    #train_size = 5000
    #test_size = 5000
    
    train_idxs = shuffled_idxs[:train_size]
    test_idxs = shuffled_idxs[train_size:train_size+test_size]
    
    # Create train and test dataloaders
    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    test_dataset = torch.utils.data.Subset(dataset, test_idxs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256)
    
    
    # Get image features for untuned AlexNet
    #alex = models.alexnet(weights='DEFAULT')
    from torchvision.models import AlexNet_Weights
    alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
    #alex = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    alex.to(device) 
    alex.eval() 
    
    #feature_extractor = create_feature_extractor(alex, return_nodes=['classifier.5']).to(device)
    #del alex
    """
    # Get untuned alexnet features
    train_features_untuned = np.zeros((train_size, 4096))
    train_labels = np.zeros(train_size)
    for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            ft = feature_extractor(data[0].to(device))
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()
        train_features_untuned[low_idx:high_idx] = ft
        train_labels[low_idx:high_idx] = data[1]
        del ft
    """
    test_features_untuned = np.zeros((test_size, 4096))
    test_labels = np.zeros(test_size)
    test_preds = np.zeros(test_size)
    for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_size = data[0].shape[0]
        if (batch_index==0):
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            output = alex(data[0].to(device))
            out_soft = torch.nn.functional.softmax(output, dim=1)
            preds = torch.argmax(out_soft, dim=1)

            #ft = feature_extractor(data[0].to(device))
        # Flatten the features
        #ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).cpu().detach().numpy()
        #test_features_untuned[low_idx:high_idx] = ft
        #del ft
        test_labels[low_idx:high_idx] = data[1]
        test_preds[low_idx:high_idx] = preds.cpu()
        
    #del feature_extractor
    print(test_labels[:20], test_preds[:20])
    print(np.count_nonzero(test_labels-test_preds==0) / test_size)
    
    #print(test_labels)
    
    scaler = StandardScaler()
    fit_scaler = scaler.fit(train_features_untuned)
    train_features_untuned = fit_scaler.transform(train_features_untuned)
    test_features_untuned = fit_scaler.transform(test_features_untuned)
    
    #classifier = LinearSVC().fit(train_features_untuned, train_labels)
    print("Fitting linear classifier...")
    classifier = LogisticRegression(max_iter=5000).fit(train_features_untuned, train_labels)
    preds = classifier.predict(test_features_untuned)
    acc = accuracy_score(test_labels, preds) * 100
    print("Untuned", acc)
    
    
    if (rois[0] == 'all'):
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
              "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA",
         "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
       "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]

    # Go through list of rois
    for roi in rois:
        
        #print(roi)
        
        # Get number of voxels for this model
        _, _, _, _, num_voxels = get_dataloaders_unshuffled(project_dir, device, subj_num, hemisphere, roi, 1024)
        
        if (num_voxels < 20):
            print(roi, "is too small or empty")
        else:

            if (tuning_method=='cl'):
                model_path = project_dir + "/cl_models/Subj" + str(subj_num) + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_model_e30.pt"
                model = torch.load(model_path).to(device)

            elif (tuning_method=='reg'):
                model_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(subj_num)
                model_path = model_dir + r"/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e75.pt" 
                model = torch.load(model_path).to(device)

            feature_extractor = tx.Extractor(model, ["alex.classifier.6"]).to(device)

            train_features_tuned = np.zeros((train_size, 1000))
            train_labels = np.zeros(train_size)
            for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    if (tuning_method=='cl'):
                        fmri_dummy = torch.zeros((batch_size, num_voxels)).to(device)
                        _, alex_out_dict = feature_extractor(fmri_dummy, data[0].to(device))
                    elif (tuning_method=='reg'):
                        _, alex_out_dict = feature_extractor(data[0].to(device))
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                train_features_tuned[low_idx:high_idx] = ft
                train_labels[low_idx:high_idx] = data[1]
                del ft

            test_features_tuned = np.zeros((test_size, 1000))
            test_labels = np.zeros(test_size)
            for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                batch_size = data[0].shape[0]
                if (batch_index==0):
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    if (tuning_method=='cl'):
                        fmri_dummy = torch.zeros((batch_size, num_voxels)).to(device)
                        _, alex_out_dict = feature_extractor(fmri_dummy, data[0].to(device))
                    elif (tuning_method=='reg'):
                        _, alex_out_dict = feature_extractor(data[0].to(device))
                ft = alex_out_dict['alex.classifier.6'].detach().cpu().numpy()
                test_features_tuned[low_idx:high_idx] = ft
                test_labels[low_idx:high_idx] = data[1]
                del ft

            scaler = StandardScaler()
            fit_scaler = scaler.fit(train_features_tuned)
            train_features_tuned = fit_scaler.transform(train_features_tuned)
            test_features_tuned = fit_scaler.transform(test_features_tuned)

            #classifier = LinearSVC().fit(train_features_tuned, train_labels)
            print("Fitting linear classifier...")
            classifier = LogisticRegression(max_iter=5000, multi_class='ovr', solver='sag', n_jobs=-1).fit(train_features_tuned, train_labels)
            preds = classifier.predict(test_features_tuned)
            acc = accuracy_score(test_labels, preds) * 100
            #results[roi] = acc
            print(roi, acc)
            
            
            if (save):
                try:
                    # Load existing results, add result for roi if not already in the existing results
                    existing_results = joblib.load(save_path)
                    if roi not in existing_results.keys():
                        existing_results[roi] = acc
                        joblib.dump(existing_results, save_path)
                except:
                    results = {}
                    results[roi] = acc
                    joblib.dump(results, save_path)
                    
