
import os
import sys
import re
import joblib
import torch
from torchvision.models import AlexNet_Weights
import torchextractor as tx
from argparse import ArgumentParser
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr as corr
from utils import get_dataloaders, get_dataloaders_cv, fit_pca
from models import CLR_model, fmri_reg, get_pooled_CL_model
import gc


# Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--root", help="Path to main folder of project", required=True)
parser.add_argument(
    "--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument(
    "--subj", help="Subject number, 1 through 8", type=int, required=True)
parser.add_argument(
    "--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--overwrite_existing", action='store_true',
                    help="Whether to overwrite already existing results")
parser.add_argument("--rois", help="List of rois", nargs='+',
                    type=str, required=False, default=[])
parser.add_argument(
    "--batch_size", help="Batch size for training", type=int, default=256)
parser.add_argument("--things", action="store_true")
parser.add_argument("--method", type=str, default="untuned",
                    help="Options are untuned, cl-tuned, reg-tuned, pooled-avg, pooled-const")
parser.add_argument("--use_test_set", action='store_true')


# Given already fit PCA, extract and compress image features
def extract_features_and_fmri(feature_extractor, dataloader, pca, num_images, num_voxels, alex_out_layer=None,
                              is_cl_feature_extractor=False, is_reg_feature_extractor=False,
                              pooled=False, subj_num=1):

    print("Extracting PCA Features...")
    features = np.zeros((num_images, 1000))
    fmri = np.zeros((num_images, num_voxels))
    for batch_index, data in enumerate(dataloader):
        if batch_index == 0:
            low_idx = 0
            high_idx = dataloader.batch_size
        else:
            low_idx = high_idx
            high_idx += dataloader.batch_size
        # Extract features
        fmri_batch = data[0]
        if is_cl_feature_extractor:
            with torch.no_grad():
                if pooled:
                    _, alex_out_dict = feature_extractor(
                        fmri_batch, data[1], subj_num)
                else:
                    _, alex_out_dict = feature_extractor(fmri_batch, data[1])
                ft = alex_out_dict[alex_out_layer].detach()
                ft = torch.flatten(ft, start_dim=1)
        elif is_reg_feature_extractor:
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[1])
            ft = alex_out_dict[alex_out_layer].detach()
            ft = torch.flatten(ft, start_dim=1)
        else:
            with torch.no_grad():
                _, alex_out_dict = feature_extractor(data[1])
                ft = alex_out_dict[alex_out_layer].detach()
                ft = torch.flatten(ft, start_dim=1)
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features[low_idx:high_idx] = ft
        fmri[low_idx:high_idx] = fmri_batch.cpu().numpy()
        del ft, fmri_batch
    return features, fmri


def main():

    args = parser.parse_args()

    # Load arguments
    project_dir = args.root
    device = args.device
    device = torch.device(device)
    subj_num = args.subj
    batch_size = args.batch_size
    hemisphere = args.hemisphere
    rois = args.rois
    method = args.method
    use_test_set = args.use_test_set
    overwrite_existing = args.overwrite_existing

    if hemisphere != 'left' and hemisphere != 'right':
        print("Invalid hemisphere, must be left or right")
        sys.exit()

    hemisphere_abbr = 'l' if hemisphere == 'left' else 'r'

    all_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies",
                "OFA", "FFA-1", "FFA-2", "mTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
                "mfs-words", "mTL-words"]

    if len(rois) == 0:
        rois = all_rois

    roi_mat_idxs_dict = {}
    for counter, roi in enumerate(all_rois):
        if hemisphere == "left":
            roi_mat_idxs_dict[roi] = counter
        else:
            roi_mat_idxs_dict[roi] = counter + len(all_rois)

    save_dir = os.path.join(
        project_dir, "encoding_results",  "Subj" + str(subj_num))
    os.chdir(save_dir)

    if method == "untuned":
        existing_results_dict_path = "best_alex_layer_dict_untuned.joblib"
        existing_results_mat_path = "best_alex_layers_mat_untuned.npy"
        best_alphas_voxel_accs_dict_path = "best_alphas_voxel_accs_dict_untuned.joblib"
        best_alphas_voxel_dict_path = "best_alphas_voxel_dict_untuned.joblib"
    elif method == "cl-tuned":
        existing_results_dict_path = "best_alex_layer_dict_cl_tuned.joblib"
        existing_results_mat_path = "best_alex_layers_mat_cl_tuned.npy"
        best_alphas_voxel_accs_dict_path = "best_alphas_voxel_accs_dict_cl_tuned.joblib"
        best_alphas_voxel_dict_path = "best_alphas_voxel_dict_cl_tuned.joblib"
    elif method == "reg-tuned":
        existing_results_dict_path = "best_alex_layer_dict_reg_tuned.joblib"
        existing_results_mat_path = "best_alex_layers_mat_reg_tuned.npy"
        best_alphas_voxel_accs_dict_path = "best_alphas_voxel_accs_dict_reg_tuned.joblib"
        best_alphas_voxel_dict_path = "best_alphas_voxel_dict_reg_tuned.joblib"
    elif method == "pooled-avg":
        existing_results_dict_path = "best_alex_layer_dict_pooled_avg.joblib"
        existing_results_mat_path = "best_alex_layers_mat_pooled_avg.npy"
        best_alphas_voxel_accs_dict_path = "best_alphas_voxel_accs_dict_pooled_avg.joblib"
        best_alphas_voxel_dict_path = "best_alphas_voxel_dict_pooled_avg.joblib"
    elif method == "pooled-const":
        existing_results_dict_path = "best_alex_layer_dict_pooled_const.joblib"
        existing_results_mat_path = "best_alex_layers_mat_pooled_const.npy"
        best_alphas_voxel_accs_dict_path = "best_alphas_voxel_accs_dict_pooled_const.joblib"
        best_alphas_voxel_dict_path = "best_alphas_voxel_dict_pooled_const.joblib"

    alex_out_layers = ["features.2", "features.5", "features.7", "features.9",
                       "features.12", "classifier.2", "classifier.5", "classifier.6"]
    alex_out_layer_dims = {"features.2": 46656, "features.5": 32448, "features.7": 64896, "features.9": 43264,
                           "features.12": 9216, "classifier.2": 4096, "classifier.5": 4096, "classifier.6": 1000}

    # Range of alpha to try for ridge regression
    alphas = list(np.logspace(-1, 7, num=9))

    for roi in rois:

        roi_idx = roi_mat_idxs_dict[roi]
        print(roi, roi_idx)

        # Skip if result already obtained
        if not overwrite_existing:
            try:
                best_alex_layers_and_alphas_dict = joblib.load(
                    existing_results_dict_path)
                hemisphere_roi_key = hemisphere_abbr + "h_" + roi
                if hemisphere_roi_key in best_alex_layers_and_alphas_dict.keys():
                    print("Already computed!")
                    return
            except:
                pass

        # Get number of voxels
        _, _, _, _, num_voxels = get_dataloaders_cv(
            project_dir, device, subj_num, hemisphere, roi, batch_size=batch_size, fold_num=0, use_test_set=use_test_set)

        if method == "untuned":
            return_layers = ["features.2", "features.5", "features.7", "features.9",
                             "features.12", "classifier.2", "classifier.5", "classifier.6"]
        else:
            return_layers = ["alex.features.2", "alex.features.5", "alex.features.7", "alex.features.9",
                             "alex.features.12", "alex.classifier.2", "alex.classifier.5", "alex.classifier.6"]

        if method == "pooled-avg" or method == "pooled-const":
            # Get list of number of voxels for each subj
            present_subjs = []
            num_voxels_subjs = []
            for subj_idx_temp in range(1, 9):
                _, _, _, _, num_voxels_temp = get_dataloaders(
                    project_dir, device, subj_idx_temp, hemisphere, roi, batch_size=batch_size)
                if num_voxels_temp != 0:
                    present_subjs.append(subj_idx_temp)
                    num_voxels_subjs.append(num_voxels_temp)

        if method == "untuned":
            # Load pretrained Alexnet model
            model = torch.hub.load(
                'pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
            model.to(device)
            model.eval()
            feature_extractor = tx.Extractor(model, return_layers)
        elif method == "cl-tuned":
            # Load CL-tuned model
            cl_model_dir = os.path.join(project_dir, "cl_models", "Subj" + str(subj_num))
            cl_model_name = "subj" + str(subj_num) + "_" + hemisphere_abbr + \
                "h_" + roi + "_model_e30.pt"
            cl_model_path = os.path.join(cl_model_dir, cl_model_name)
            h_dim = int(num_voxels*0.8)
            z_dim = int(num_voxels*0.2)
            model = CLR_model(num_voxels, h_dim, z_dim)
            # Some models are saved differently
            try:
                model.load_state_dict(torch.load(
                    cl_model_path, map_location=device, weights_only=False)[0].state_dict())
            except:
                try:
                    model.load_state_dict(torch.load(
                        cl_model_path, map_location=device, weights_only=False).state_dict())
                except:
                    model.load_state_dict(torch.load(
                        cl_model_path, map_location=device, weights_only=False))
            model.to(device)
            model.eval()
            feature_extractor = tx.Extractor(model, return_layers)
        elif method == "reg-tuned":
            # Load regression-tuned model
            reg_model_dir = os.path.join(project_dir, "baseline_models", "nn_reg", "Subj" + str(subj_num))
            reg_model_name = "subj" + str(subj_num) + "_" + hemisphere_abbr + \
                "h_" + roi + "_reg_model_e75.pt"
            reg_model_path = os.path.join(reg_model_dir, reg_model_name)
            model = fmri_reg(num_voxels)
            # Some models are saved differently
            try:
                model.load_state_dict(torch.load(reg_model_path, map_location=torch.device(
                    'cpu'), weights_only=False)[0].state_dict())
            except:
                try:
                    model.load_state_dict(torch.load(reg_model_path, map_location=torch.device(
                        'cpu'), weights_only=False).state_dict())
                except:
                    model.load_state_dict(torch.load(
                        reg_model_path, map_location=torch.device('cpu'), weights_only=False))
            model.to(device)
            model.eval()
            feature_extractor = tx.Extractor(model, return_layers)
        elif method == "pooled-avg":
            # Load pooled (h-avg) model
            avg_voxel_dim = int(np.mean(np.array(num_voxels_subjs)))
            model, _ = get_pooled_CL_model(
                num_voxels_subjs, device, avg_voxel_dim)
            model_path = os.path.join(
                project_dir, "cl_models", hemisphere_abbr + "h_" + roi + "_pooled_model_e30_havg.pt")
            # Some models are saved differently
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device(
                    'cpu'), weights_only=False)[0].state_dict())
            except:
                try:
                    model.load_state_dict(torch.load(model_path, map_location=torch.device(
                        'cpu'), weights_only=False).state_dict())
                except:
                    model.load_state_dict(torch.load(
                        model_path, map_location=torch.device('cpu'), weights_only=False))
            model.to(device)
            model.eval()
            feature_extractor = tx.Extractor(model, return_layers)
            subj_num_adjusted = present_subjs.index(subj_num) + 1
        elif method == "pooled-const":
            # Load pooled (h-const) model
            h_max_dim = 5741
            model, _ = get_pooled_CL_model(num_voxels_subjs, device, h_max_dim)
            model_path = os.path.join(
                project_dir, "cl_models", hemisphere_abbr + "h_" + roi + "_pooled_model_e30_hconst.pt")
            # Some models are saved differently
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device(
                    'cpu'), weights_only=False)[0].state_dict())
            except:
                try:
                    model.load_state_dict(torch.load(model_path, map_location=torch.device(
                        'cpu'), weights_only=False).state_dict())
                except:
                    model.load_state_dict(torch.load(
                        model_path, map_location=torch.device('cpu'), weights_only=False))
            model.to(device)
            model.eval()
            feature_extractor = tx.Extractor(model, return_layers)
            subj_num_adjusted = present_subjs.index(subj_num) + 1

        fold_accs_dict = {"features.2": 0, "features.5": 0, "features.7": 0, "features.9": 0,
                          "features.12": 0, "classifier.2": 0, "classifier.5": 0, "classifier.6": 0}
        # Store the val acc for each voxel for each alpha, layer (sum over folds)
        fold_voxel_accs_dict = {"features.2": [np.zeros(num_voxels) for i in range(len(alphas))], "features.5": [np.zeros(num_voxels) for i in range(len(alphas))], "features.7": [np.zeros(num_voxels) for i in range(len(alphas))], "features.9": [np.zeros(num_voxels) for i in range(len(
            alphas))], "features.12": [np.zeros(num_voxels) for i in range(len(alphas))], "classifier.2": [np.zeros(num_voxels) for i in range(len(alphas))], "classifier.5": [np.zeros(num_voxels) for i in range(len(alphas))], "classifier.6": [np.zeros(num_voxels) for i in range(len(alphas))]}
        fold_alphas_dict = {"features.2": [], "features.5": [], "features.7": [], "features.9": [
        ], "features.12": [], "classifier.2": [], "classifier.5": [], "classifier.6": []}

        for fold_num in range(5):

            # Get dataloaders
            train_dataloader, val_dataloader, train_size, val_size, num_voxels = get_dataloaders_cv(
                project_dir, device, subj_num, hemisphere, roi, batch_size=batch_size, fold_num=fold_num, use_test_set=use_test_set)

            # Check if this ROI is empty
            print(num_voxels)
            if num_voxels == 0:
                print("Empty")
                try:
                    best_alex_layers_and_alphas_dict = joblib.load(
                        existing_results_dict_path)
                except:
                    best_alex_layers_and_alphas_dict = {}
                best_alex_layers_and_alphas_dict[hemisphere_abbr +
                                                 "h_" + roi] = "Empty"
                break

            # Keep track of accuracy (using best alpha for each voxel) for each layer (and acc for each voxel)
            layer_accs_dict = {"features.2": 0, "features.5": 0, "features.7": 0, "features.9": 0,
                               "features.12": 0, "classifier.2": 0, "classifier.5": 0, "classifier.6": 0}
            layer_voxel_accs_dict = {"features.2": np.zeros(num_voxels), "features.5": np.zeros(num_voxels), "features.7": np.zeros(num_voxels), "features.9": np.zeros(num_voxels),
                                     "features.12": np.zeros(num_voxels), "classifier.2": np.zeros(num_voxels), "classifier.5": np.zeros(num_voxels), "classifier.6": np.zeros(num_voxels)}
            layer_alphas_dict = {"features.2": np.zeros(num_voxels), "features.5": np.zeros(num_voxels), "features.7": np.zeros(num_voxels), "features.9": np.zeros(num_voxels),
                                 "features.12": np.zeros(num_voxels), "classifier.2": np.zeros(num_voxels), "classifier.5": np.zeros(num_voxels), "classifier.6": np.zeros(num_voxels)}

            for layer_idx, layer in enumerate(alex_out_layers):

                print("Fold " + str(fold_num) + ", layer " + layer + ": \n")

                # Fit and extract image features from PCA
                if method == "cl-tuned":
                    cl_layer = "alex." + layer
                    pca = fit_pca(feature_extractor, train_dataloader, train_size, cl_layer,
                                  alex_out_layer_dims[layer], is_cl_feature_extractor=True, make_dummy_fmri_data=True, num_voxels=num_voxels)
                    train_features, train_fmri = extract_features_and_fmri(feature_extractor, train_dataloader, pca, train_size, num_voxels,
                                                                           alex_out_layer=cl_layer, is_cl_feature_extractor=True)
                    val_features, val_fmri = extract_features_and_fmri(feature_extractor, val_dataloader, pca, val_size, num_voxels,
                                                                       alex_out_layer=cl_layer, is_cl_feature_extractor=True)
                elif method == "pooled-avg" or method == "pooled-const":
                    cl_layer = "alex." + layer
                    pca = fit_pca(feature_extractor, train_dataloader, train_size, cl_layer, alex_out_layer_dims[layer], is_cl_feature_extractor=True, make_dummy_fmri_data=True,
                                  num_voxels=num_voxels, pooled=True, subj_num=subj_num_adjusted)
                    train_features, train_fmri = extract_features_and_fmri(feature_extractor, train_dataloader, pca, train_size, num_voxels,
                                                                           alex_out_layer=cl_layer, is_cl_feature_extractor=True, pooled=True, subj_num=subj_num_adjusted)
                    val_features, val_fmri = extract_features_and_fmri(feature_extractor, val_dataloader, pca, val_size, num_voxels,
                                                                       alex_out_layer=cl_layer, is_cl_feature_extractor=True, pooled=True, subj_num=subj_num_adjusted)
                elif method == "reg-tuned":
                    reg_layer = "alex." + layer
                    pca = fit_pca(feature_extractor, train_dataloader, train_size, reg_layer,
                                  alex_out_layer_dims[layer], is_reg_feature_extractor=True, make_dummy_fmri_data=True, num_voxels=num_voxels)
                    train_features, train_fmri = extract_features_and_fmri(feature_extractor, train_dataloader, pca, train_size, num_voxels, alex_out_layer=reg_layer,
                                                                           is_reg_feature_extractor=True)
                    val_features, val_fmri = extract_features_and_fmri(feature_extractor, val_dataloader, pca, val_size, num_voxels, alex_out_layer=reg_layer,
                                                                       is_reg_feature_extractor=True)
                else:
                    pca = fit_pca(feature_extractor, train_dataloader,
                                  train_size, layer, alex_out_layer_dims[layer])
                    train_features, train_fmri = extract_features_and_fmri(
                        feature_extractor, train_dataloader, pca, train_size, num_voxels, alex_out_layer=layer)
                    val_features, val_fmri = extract_features_and_fmri(
                        feature_extractor, val_dataloader, pca, val_size, num_voxels, alex_out_layer=layer)

                # Find best alpha for each voxel using this layer, accuracy with that alpha
                voxels_best_alphas = np.zeros(num_voxels)
                voxels_best_corrs = np.zeros(num_voxels)
                for alpha_idx, alpha in enumerate(alphas):

                    encoding_model = Ridge(alpha=alpha, random_state=0).fit(
                        train_features, train_fmri)
                    preds = encoding_model.predict(val_features)

                    corrs = np.zeros(num_voxels)
                    for v in range(num_voxels):
                        corrs[v] = corr(val_fmri[:, v], preds[:, v])[0]
                        if corrs[v] > voxels_best_corrs[v]:
                            voxels_best_corrs[v] = corrs[v]
                            voxels_best_alphas[v] = alpha
                    fold_voxel_accs_dict[layer][alpha_idx] = fold_voxel_accs_dict[layer][alpha_idx] + corrs

                # Acc for this layer on this fold using best alpha for each voxel
                layer_avg_acc = voxels_best_corrs.mean()
                print(roi + " layer:" + layer + " acc: " + str(layer_avg_acc))
                layer_alphas_dict[layer] = voxels_best_alphas
                layer_accs_dict[layer] = layer_avg_acc
                layer_voxel_accs_dict[layer] = voxels_best_corrs

                del pca, train_features, train_fmri, val_features, val_fmri, encoding_model, preds
                gc.collect()

            # Running average accuracy across folds, accumulate alphas
            for layer in fold_accs_dict.keys():
                fold_accs_dict[layer] = fold_accs_dict[layer] + \
                    layer_accs_dict[layer]  # / (fold_num + 1)
                fold_alphas_dict[layer].append(layer_alphas_dict[layer])

            del train_dataloader, val_dataloader, layer_alphas_dict
            gc.collect()

        # Average over 5 folds
        for layer in fold_accs_dict.keys():
            fold_accs_dict[layer] = fold_accs_dict[layer] / 5
            for alpha_idx, alpha in enumerate(alphas):
                fold_voxel_accs_dict[layer][alpha_idx] = fold_voxel_accs_dict[layer][alpha_idx] / 5

        # Get results
        if num_voxels != 0:
            # Go through each layer, get the best alpha for each voxel, and the voxel acc for that alpha
            layer_overall_accs = {"features.2": 0, "features.5": 0, "features.7": 0, "features.9": 0,
                                  "features.12": 0, "classifier.2": 0, "classifier.5": 0, "classifier.6": 0}
            layer_best_alphas = {"features.2": np.zeros(num_voxels), "features.5": np.zeros(num_voxels), "features.7": np.zeros(num_voxels), "features.9": np.zeros(
                num_voxels), "features.12": np.zeros(num_voxels), "classifier.2": np.zeros(num_voxels), "classifier.5": np.zeros(num_voxels), "classifier.6": np.zeros(num_voxels)}
            layer_best_alphas_voxel_accs = {"features.2": np.zeros(num_voxels), "features.5": np.zeros(num_voxels), "features.7": np.zeros(num_voxels), "features.9": np.zeros(
                num_voxels), "features.12": np.zeros(num_voxels), "classifier.2": np.zeros(num_voxels), "classifier.5": np.zeros(num_voxels), "classifier.6": np.zeros(num_voxels)}
            best_layer = None
            best_layer_acc = -1
            for layer in fold_voxel_accs_dict.keys():
                for voxel_idx in range(num_voxels):
                    best_alpha = -1
                    best_acc = -1
                    for alpha_idx in range(len(alphas)):
                        voxel_alpha_acc = fold_voxel_accs_dict[layer][alpha_idx][voxel_idx]
                        if voxel_alpha_acc > best_acc:
                            best_acc = voxel_alpha_acc
                            best_alpha = alphas[alpha_idx]
                    layer_best_alphas[layer][voxel_idx] = best_alpha
                    layer_best_alphas_voxel_accs[layer][voxel_idx] = best_acc
                layer_acc = np.mean(layer_best_alphas_voxel_accs[layer])
                layer_overall_accs[layer] = layer_acc
                if layer_acc > best_layer_acc:
                    best_layer = layer
                    best_layer_acc = layer_acc

            # Load existing results
            try:
                best_alex_layers_and_alphas_dict = joblib.load(
                    existing_results_dict_path)
            except:
                best_alex_layers_and_alphas_dict = {}
            try:
                results_mat = np.load(existing_results_mat_path)
            except:
                # Size is ROIs (both hemisphers) by layers, stores avg acc for each layer,roi (using best alpha for each voxel)
                results_mat = np.zeros((len(all_rois)*2, 8))
            best_alex_layers_and_alphas_dict[hemisphere_abbr + "h_" + roi] = (
                best_layer, layer_best_alphas[best_layer])
            for layer_idx, (layer, layer_acc) in enumerate(zip(layer_overall_accs.keys(), layer_overall_accs.values())):
                print(layer, layer_acc)
                results_mat[roi_idx, layer_idx] = layer_acc

            # Save best layers after each ROI, results for each voxel
            joblib.dump(best_alex_layers_and_alphas_dict,
                        existing_results_dict_path)
            np.save(existing_results_mat_path, results_mat)

            try:
                best_alphas_voxel_accs_dict = joblib.load(
                    best_alphas_voxel_accs_dict_path)
            except:
                best_alphas_voxel_accs_dict = {}
            try:
                best_alphas_voxel_dict = joblib.load(
                    best_alphas_voxel_dict_path)
            except:
                best_alphas_voxel_dict = {}
            best_alphas_voxel_accs_dict[hemisphere_abbr + "h_" +
                                        roi] = layer_best_alphas_voxel_accs[best_layer]
            best_alphas_voxel_dict[hemisphere_abbr +
                                   "h_" + roi] = layer_best_alphas[best_layer]
            joblib.dump(best_alphas_voxel_accs_dict,
                        best_alphas_voxel_accs_dict_path)
            joblib.dump(best_alphas_voxel_dict, best_alphas_voxel_dict_path)


if __name__ == '__main__':
    main()
