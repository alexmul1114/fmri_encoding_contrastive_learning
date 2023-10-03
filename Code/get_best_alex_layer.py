# Imports

import os
import sys

import joblib

import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import AlexNet_Weights

from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr as corr

from statistics import mode

# Local imports
from utils import get_dataloaders_cv, get_roi_mapping_files, fit_pca


# Command line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to main folder of project", required=True)
parser.add_argument("--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument("--subj", help="Subject number, 1 through 8", type=int, required=True)
parser.add_argument("--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--overwrite-existing", help="Whether to overwrite already existing results", type=bool, default=False)
parser.add_argument("--rois", help="List of rois", nargs='+', type=str, required=False, default=[])
parser.add_argument("--batch-size", help="Batch size for training", type=int, default=1024)



# Given already fit PCA, extract and compress image features
def extract_features_and_fmri(feature_extractor, dataloader, pca, num_images, num_voxels):

    print("Extracting PCA Features...")
    features = np.zeros((num_images, 1000))
    fmri = np.zeros((num_images, num_voxels))
    for batch_index, data in enumerate(dataloader):
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
    rois = args.rois
    overwrite_existing = args.overwrite_existing
    
    if (hemisphere != 'left' and hemisphere != 'right'):
        print("Invalid hemisphere, must be left or right")
        sys.exit()
        
    hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
        
    rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1",
            "FFA-2", "mTL-faces", "aTL-faces", "OPA",
              "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
           "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal", "All vertices"]
    
    roi_map_dir = project_dir + r"/roi_masks/Subj" + str(subj_num)
    save_dir = project_dir + r"/best_alex_out_layers/Subj" + str(subj_num) 
    os.chdir(save_dir)
    
    # Dict to keep track of best layers and corresponding best alphas for each ROI (key = roi name, value = (best_layer, best_alpha))
    if (overwrite_existing):
        best_alex_layers_and_alphas_dict = {}
    else:
        existing_results_path = project_dir + r"/best_alex_out_layers/Subj" + str(subj_num) + "/subj" + str(subj_num) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
        best_alex_layers_and_alphas_dict = joblib.load(existing_results_path)
        existing_roi_results = best_alex_layers_and_alphas_dict.keys()
        for roi in existing_roi_results:
            all_rois.remove(roi)

    alex_out_layers = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"]
    alex_out_layer_dims = {"features.2":46656, "features.5":32448, "features.7":64896, "features.9":43264, "features.12":9216, "classifier.2":4096, "classifier.5":4096, "classifier.6":1000}
    
    # Range of alpha to try for ridge regression
    alphas = list(np.logspace(-1, 7, num=9))
    
    for roi in rois:
         
        fold_accs_dict = {"features.2":0, "features.5":0, "features.7":0, "features.9":0, "features.12":0, "classifier.2":0, "classifier.5":0, "classifier.6":0}
        fold_alphas_dict = {"features.2":[], "features.5":[], "features.7":[], "features.9":[], "features.12":[], "classifier.2":[], "classifier.5":[], "classifier.6":[]}
    
        for fold_num in range(5):

            # Get dataloaders 
            train_dataloader, val_dataloader, train_size, val_size, num_voxels = get_dataloaders_cv(project_dir, device, subj_num, hemisphere, roi, batch_size=1024, fold_num=fold_num)
            
            # Check if this ROI is empty
            if (num_voxels==0):
                best_alex_layers_and_alphas_dict[roi] = "Empty"
                break
                
            # Keep track of accuracy (using best alpha) for each layer
            layer_accs_dict = {"features.2":0, "features.5":0, "features.7":0, "features.9":0, "features.12":0, "classifier.2":0, "classifier.5":0, "classifier.6":0}
            layer_alphas_dict = {"features.2":0, "features.5":0, "features.7":0, "features.9":0, "features.12":0, "classifier.2":0, "classifier.5":0, "classifier.6":0}

            for layer in alex_out_layers:

                # Load pretrained Alexnet model
                model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
                model.to(device) 
                model.eval()
                feature_extractor = create_feature_extractor(model, return_nodes=[layer])

                # Fit and extract image features from PCA
                pca = fit_pca(feature_extractor, train_dataloader, train_size, alex_out_layer_dims[layer], batch_size=1024)
                train_features, train_fmri = extract_features_and_fmri(feature_extractor, train_dataloader, pca, train_size, num_voxels)
                val_features, val_fmri = extract_features_and_fmri(feature_extractor, val_dataloader, pca, val_size, num_voxels)

                
                # Find best alpha using this layer, accuracy with that alpha
                results_dict = dict(zip(alphas, [0]*len(alphas)))
                for alpha in alphas:
                    
                    encoding_model = Ridge(alpha=alpha).fit(train_features, train_fmri)
                    preds = encoding_model.predict(val_features)
                    
                    corrs = np.zeros(num_voxels)
                    for v in range(num_voxels):
                        corrs[v] = corr(val_fmri[:, v], preds[:, v])[0]

                    avg_acc = corrs.mean()
                    results_dict[alpha] = avg_acc
                print(results_dict)
                best_alpha = max(zip(results_dict.values(), results_dict.keys()))[1]
                best_acc = max(zip(results_dict.values(), results_dict.keys()))[0]
                print(roi + " best alpha for layer:" + layer + ": " + str(best_alpha) + " with acc: " + str(best_acc))
                layer_alphas_dict[layer] = best_alpha
                layer_accs_dict[layer] = best_acc
                
            # Running average accuracy across folds, accumulate alphas
            for layer in fold_accs_dict.keys():
                fold_accs_dict[layer] = (fold_accs_dict[layer] + layer_accs_dict[layer]) / (fold_num + 1)
                fold_alphas_dict[layer].append(layer_alphas_dict[layer])
            
                
                
        # Get layer which had highest accuracy  
        if (num_voxels != 0):
            best_layer = max(zip(fold_accs_dict.values(), fold_accs_dict.keys()))[1]
            best_acc = max(zip(fold_accs_dict.values(), fold_accs_dict.keys()))[0]
            print("Alphas for best layer:", fold_alphas_dict[best_layer])
            best_alpha = mode(fold_alphas_dict[best_layer])
            print("Chosen alpha:", best_alpha)


            best_alex_layers_and_alphas_dict[roi] = (best_layer, best_alpha)


        # Save best layers after each ROI
        save_file_name = "subj" + str(subj_num) + "_" + hemisphere_abbr + "h_best_alex_layer_dict.joblib"
        #joblib.dump(best_alex_layers_and_alphas_dict, save_file_name)
