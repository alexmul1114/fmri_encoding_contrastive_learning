
from models import CLR_model, fmri_reg, get_pooled_CL_model
from utils import get_dataloaders
from sklearn.metrics.pairwise import cosine_similarity
import torchextractor as tx
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import AlexNet_Weights
import torchvision
import joblib
from torchvision.models import AlexNet_Weights
from torchvision import transforms
import torch
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'


# Get results for image classification task
def image_classification_results(project_dir, subj_num, hemisphere, rois, device, tuning_method='cl', dataset_name='caltech256', save=False,
                                 pooled=False, pooled_h_method='const', save_probs=False):

    hemisphere_abbr = 'l' if hemisphere == 'left' else 'r'

    if pooled and pooled_h_method == 'avg':
        save_path = os.path.join(project_dir, "results", hemisphere_abbr + "h_" +
                                 dataset_name + "_" + tuning_method + "_results_pooled_havg.joblib")
        if save_probs:
            probs_save_folder = os.path.join(
                project_dir, "classification_preds", "Pooled")
    elif pooled and pooled_h_method == 'const':
        save_path = os.path.join(project_dir, "results", hemisphere_abbr + "h_" +
                                 dataset_name + "_" + tuning_method + "_results_pooled_hconst.joblib")
        if save_probs:
            probs_save_folder = os.path.join(
                project_dir, "classification_preds", "Pooled")
    else:
        save_path = project_dir + "/results/Subj" + str(subj_num) + "/subj" + str(
            subj_num) + "_" + hemisphere_abbr + "h_" + dataset_name + "_" + tuning_method + "_results.joblib"
        probs_save_folder = os.path.join(
            project_dir, "classification_preds", "Subj" + str(subj_num))

    # Seed RNG, define image transforms for alexnet
    torch.manual_seed(0)
    alex_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # convert the images to a PyTorch tensor
        # normalize the images color channels
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the Data
    if (dataset_name == 'caltech256'):
        image_dir = os.path.join(project_dir, "caltech256")
        dataset = torchvision.datasets.ImageFolder(
            root=image_dir, transform=alex_transform)
    elif (dataset_name == 'places365'):
        image_dir = os.path.join(project_dir, "places365")
        dataset = torchvision.datasets.Places365(
            root=image_dir, split='val', small=True, transform=alex_transform)
    elif (dataset_name == 'sun397'):
        image_dir = os.path.join(project_dir, "sun397")
        dataset = torchvision.datasets.SUN397(
            root=image_dir, transform=alex_transform, download=True)
    elif (dataset_name == 'imagenet'):
        image_dir = os.path.join(project_dir, "imagenet")
        dataset = torchvision.datasets.ImageNet(
            root=image_dir, split='val', transform=alex_transform)

    total_num_images = len(dataset)
    generator = torch.Generator()
    generator.manual_seed(0)
    shuffled_idxs = torch.randperm(total_num_images, generator=generator)

    train_size = int(0.85 * total_num_images)
    test_size = total_num_images - train_size
    print(train_size, test_size)

    train_idxs = shuffled_idxs[:train_size]
    test_idxs = shuffled_idxs[train_size:train_size+test_size]

    # Create train and test dataloaders
    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    test_dataset = torch.utils.data.Subset(dataset, test_idxs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False)

    # Get image features for untuned AlexNet
    alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet',
                          weights=AlexNet_Weights.IMAGENET1K_V1)
    alex.to(device)
    alex.eval()

    feature_extractor = create_feature_extractor(
        alex, return_nodes=['classifier.5']).to(device)
    del alex

    # Get untuned alexnet features
    train_features_untuned = np.zeros((train_size, 4096))
    train_labels = np.zeros(train_size)
    for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_size = data[0].shape[0]
        if batch_index == 0:
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            ft = feature_extractor(data[0].to(device))
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1)
                          for l in ft.values()]).cpu().detach().numpy()
        train_features_untuned[low_idx:high_idx] = ft
        train_labels[low_idx:high_idx] = data[1]
        del ft

    test_features_untuned = np.zeros((test_size, 4096))
    test_labels = np.zeros(test_size)
    for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_size = data[0].shape[0]
        if batch_index == 0:
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            ft = feature_extractor(data[0].to(device))
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1)
                          for l in ft.values()]).cpu().detach().numpy()
        test_features_untuned[low_idx:high_idx] = ft
        del ft
        test_labels[low_idx:high_idx] = data[1]

    del feature_extractor

    # Save labels
    if save_probs:
        np.save(os.path.join(project_dir, "classification_preds",
                dataset_name + "_test_labels.npy"), test_labels)

    scaler = StandardScaler()
    fit_scaler = scaler.fit(train_features_untuned)
    train_features_untuned = fit_scaler.transform(train_features_untuned)
    test_features_untuned = fit_scaler.transform(test_features_untuned)

    print("Fitting linear classifier...")
    classifier = LogisticRegression(max_iter=5000).fit(
        train_features_untuned, train_labels)
    preds = classifier.predict(test_features_untuned)
    if save_probs:
        untuned_pred_probs = classifier.predict_proba(test_features_untuned)
        untuned_pred_probs_save_path = os.path.join(
            project_dir, "classification_preds", "Untuned", "untuned_" + dataset_name + "_test_probs.npy")
        np.save(untuned_pred_probs_save_path, untuned_pred_probs)
    acc = accuracy_score(test_labels, preds) * 100
    print("Untuned", acc)

    if rois[0] == 'all' and hemisphere == 'left':
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
                "OFA", "FFA-1", "FFA-2", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words"]
    elif rois[0] == 'all' and hemisphere == 'right':
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
                "mTL-bodies", "OFA", "FFA-1", "FFA-2", "OPA",
                "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]

    # Go through list of rois
    for roi in rois:

        print(roi)
        found_subj = False

        # Get number of voxels for this model
        _, _, _, _, num_voxels = get_dataloaders(
            project_dir, device, subj_num, hemisphere, roi, 1024, shuffle=False)

        if num_voxels < 20 and not pooled:
            print(roi, "is too small or empty")
        elif num_voxels >= 20 and not pooled:
            found_subj = True

        # If using pooled models, may need dummy fmri input data matching shape of some other subject's acitvations
        elif num_voxels >= 20 and pooled:
            found_subj = True
        elif num_voxels < 20 and pooled:
            for subj_idx in range(2, 9):
                _, _, _, _, num_voxels = get_dataloaders(
                    project_dir, device, subj_idx, hemisphere, roi, 1024, shuffle=False)
                if num_voxels > 0:
                    found_subj = True
                    break

        if found_subj:
            if tuning_method == 'cl':
                if pooled == True:
                    # Get voxel counts for ROI across all subjects with the ROI from previously created dictionary
                    voxel_counts_file = os.path.join(
                        project_dir, hemisphere_abbr + "h_voxel_counts_rois.joblib")
                    voxel_counts = joblib.load(voxel_counts_file)
                    num_voxels_subjs = voxel_counts[roi]
                    avg_voxel_dim = int(np.mean(np.array(num_voxels_subjs)))
                    if pooled_h_method == 'avg':
                        # Load pooled CL model
                        model, _ = get_pooled_CL_model(
                            num_voxels_subjs, device, avg_voxel_dim)
                        model_path = os.path.join(
                            project_dir, "cl_models", hemisphere_abbr + "h_" + roi + "_pooled_model_e30_havg.pt")
                    elif pooled_h_method == 'const':
                        h_max_dim = 5741
                        # Load pooled model
                        model, _ = get_pooled_CL_model(
                            num_voxels_subjs, device, h_max_dim)
                        model_path = os.path.join(
                            project_dir, "cl_models", hemisphere_abbr + "h_" + roi + "_pooled_model_e30_hconst.pt")
                else:
                    model_dir = os.path.join(project_dir, "cl_models", "Subj" + str(subj_num))
                    model_path = os.path.join(model_dir, "subj" + \
                        str(subj_num) + "_" + hemisphere_abbr + \
                        "h_" + roi + "_model_e30.pt")
                    h_dim = int(num_voxels*0.8)
                    z_dim = int(num_voxels*0.2)
                    model = CLR_model(num_voxels, h_dim, z_dim)

            elif tuning_method == 'reg':
                model_dir = os.path.join(project_dir, "baseline_models", "nn_reg", "Subj" + str(subj_num))
                model_path = os.path.join(model_dir, "subj" + \
                    str(subj_num) + "_" + hemisphere_abbr + \
                    "h_" + roi + "_reg_model_e75.pt")
                model = fmri_reg(num_voxels)

            # Some models are saved differently
            try:
                model.load_state_dict(torch.load(
                    model_path, map_location=torch.device('cpu'))[0].state_dict())
            except:
                try:
                    model.load_state_dict(torch.load(
                        model_path, map_location=torch.device('cpu')).state_dict())
                except:
                    model.load_state_dict(torch.load(
                        model_path, map_location=torch.device('cpu')))

            model.to(device)
            model.eval()
            feature_extractor = tx.Extractor(
                model, ["alex.classifier.5"]).to(device)

            train_features_tuned = np.zeros((train_size, 4096))
            train_labels = np.zeros(train_size)
            for batch_index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_size = data[0].shape[0]
                if batch_index == 0:
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    if tuning_method == 'cl':
                        fmri_dummy = torch.zeros(
                            (batch_size, num_voxels)).to(device)
                        if pooled:
                            _, alex_out_dict = feature_extractor(
                                fmri_dummy, data[0], subj_num)
                        else:
                            _, alex_out_dict = feature_extractor(
                                fmri_dummy, data[0])
                        # _, alex_out_dict = feature_extractor(fmri_dummy, data[0].to(device))
                    elif tuning_method == 'reg':
                        _, alex_out_dict = feature_extractor(
                            data[0].to(device))
                ft = alex_out_dict['alex.classifier.5'].detach().cpu().numpy()
                train_features_tuned[low_idx:high_idx] = ft
                train_labels[low_idx:high_idx] = data[1]
                del ft

            test_features_tuned = np.zeros((test_size, 4096))
            test_labels = np.zeros(test_size)
            for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                batch_size = data[0].shape[0]
                if batch_index == 0:
                    low_idx = 0
                    high_idx = batch_size
                else:
                    low_idx = high_idx
                    high_idx += batch_size
                # Extract features
                with torch.no_grad():
                    if tuning_method == 'cl':
                        fmri_dummy = torch.zeros(
                            (batch_size, num_voxels)).to(device)
                        if pooled:
                            _, alex_out_dict = feature_extractor(
                                fmri_dummy, data[0].to(torch.float), subj_num)
                        else:
                            _, alex_out_dict = feature_extractor(
                                fmri_dummy, data[0])
                        # _, alex_out_dict = feature_extractor(fmri_dummy, data[0].to(device))
                    elif tuning_method == 'reg':
                        _, alex_out_dict = feature_extractor(
                            data[0].to(device))
                ft = alex_out_dict['alex.classifier.5'].detach().cpu().numpy()
                test_features_tuned[low_idx:high_idx] = ft
                test_labels[low_idx:high_idx] = data[1]
                del ft

            scaler = StandardScaler()
            fit_scaler = scaler.fit(train_features_tuned)
            train_features_tuned = fit_scaler.transform(train_features_tuned)
            test_features_tuned = fit_scaler.transform(test_features_tuned)

            print("Fitting linear classifier...")
            classifier = LogisticRegression(max_iter=5000).fit(
                train_features_tuned, train_labels)
            preds = classifier.predict(test_features_tuned)
            if save_probs:
                tuned_pred_probs = classifier.predict_proba(
                    test_features_tuned)
                if pooled:
                    if pooled_h_method == 'avg':
                        tuned_pred_probs_save_path = os.path.join(
                            probs_save_folder, hemisphere_abbr + "h_" + roi + "_pooled_havg_" + dataset_name + "_test_probs.npy")
                    elif pooled_h_method == 'const':
                        tuned_pred_probs_save_path = os.path.join(
                            probs_save_folder, hemisphere_abbr + "h_" + roi + "_pooled_hconst_" + dataset_name + "_test_probs.npy")
                else:
                    if tuning_method == 'cl':
                        tuned_pred_probs_save_path = os.path.join(probs_save_folder, "subj" + str(
                            subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_" + dataset_name + "_test_probs.npy")
                    else:
                        tuned_pred_probs_save_path = os.path.join(probs_save_folder, "subj" + str(
                            subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_" + dataset_name + "_reg_test_probs.npy")
                np.save(tuned_pred_probs_save_path, tuned_pred_probs)
            acc = accuracy_score(test_labels, preds) * 100
            # results[roi] = acc
            print(roi, acc)

            if save:
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


# Compute lower bound on mutual information between CNN features and ROI response using the testing data
def compute_mi_lower_bound(project_dir, device, subj_num, roi, hemisphere, pooled=False, h_method='const'):

    # Temperature tau used to tune CL models
    tau = 0.3

    hemisphere_abbr = 'l' if hemisphere == 'left' else 'r'
    print(roi, hemisphere)
    # Get test dataloader
    _, test_dataloader, _, test_size, num_voxels = get_dataloaders(
        project_dir, device, subj_num, hemisphere, roi, 1024, shuffle=False)

    if (num_voxels == 0):
        print("Empty ROI")
        return -1, -1, -1, -1, -1, -1, -1
    elif (num_voxels < 20):
        print("Too few voxels")
        return -1, -1, -1, -1, -1, -1, -1

    if pooled:
        # Get list of number of voxels for each subj
        present_subjs = []
        num_voxels_subjs = []
        for subj_idx in range(1, 9):
            _, _, _, _, num_voxels = get_dataloaders(
                project_dir, device, subj_idx, hemisphere, roi, batch_size=1024)
            if num_voxels != 0:
                present_subjs.append(subj_idx)
                num_voxels_subjs.append(num_voxels)

        # Get voxel counts for ROI across all subjects with the ROI from previously created dictionary
        voxel_counts_file = os.path.join(
            project_dir, hemisphere_abbr + "h_voxel_counts_rois.joblib")
        voxel_counts = joblib.load(voxel_counts_file)
        num_voxels_subjs = voxel_counts[roi]
        avg_voxel_dim = int(np.mean(np.array(num_voxels_subjs)))

        # Get pooled model
        print("Getting CL predictions...")
        cl_model_dir = os.path.join(project_dir, "cl_models")
        if h_method == 'avg':
            h_dim = avg_voxel_dim
            # Load pooled CL model
            cl_model, _ = get_pooled_CL_model(num_voxels_subjs, device, h_dim)
            cl_model_path = os.path.join(
                cl_model_dir, hemisphere_abbr + "h_" + roi + "_pooled_model_e30_havg.pt")
        elif h_method == 'const':
            h_dim = 5741
            # Load pooled model
            cl_model, _ = get_pooled_CL_model(num_voxels_subjs, device, h_dim)
            cl_model_path = os.path.join(
                cl_model_dir, hemisphere_abbr + "h_" + roi + "_pooled_model_e30_hconst.pt")
        z_dim = int(h_dim * 0.25)

        # Some models are saved differently
        try:
            cl_model.load_state_dict(torch.load(
                cl_model_path, map_location=torch.device('cpu'))[0].state_dict())
        except:
            try:
                cl_model.load_state_dict(torch.load(
                    cl_model_path, map_location=torch.device('cpu')).state_dict())
            except:
                cl_model.load_state_dict(torch.load(
                    cl_model_path, map_location=torch.device('cpu')))
    else:
        # Load CL-tuned model
        cl_model_dir = os.path.join(project_dir, "Subj" + str(subj_num))
        cl_model_path = os.path.join(cl_model_dir, "subj" + str(subj_num) + "_" + hemisphere_abbr +
                                     "h_" + roi + "_model_e30.pt")
        h_dim = int(num_voxels*0.8)
        z_dim = int(num_voxels*0.2)
        cl_model = CLR_model(num_voxels, h_dim, z_dim)
        # Some models are saved differently
        try:
            cl_model.load_state_dict(torch.load(
                cl_model_path, map_location=torch.device('cpu'))[0].state_dict())
        except:
            try:
                cl_model.load_state_dict(torch.load(
                    cl_model_path, map_location=torch.device('cpu')).state_dict())
            except:
                cl_model.load_state_dict(torch.load(
                    cl_model_path, map_location=torch.device('cpu')))
    cl_model.to(device)
    cl_model.eval()

    # Use just 1 batch
    K = test_size
    # Get outputs (after projection head) for CL model and fMRI data
    nn_z = np.zeros((K, z_dim))
    fmri_z = np.zeros((K, z_dim))
    for batch_index, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_size = data[0].shape[0]
        if batch_index == 0:
            low_idx = 0
            high_idx = batch_size
        else:
            low_idx = high_idx
            high_idx += batch_size
        # Extract features
        with torch.no_grad():
            if pooled:
                # Forward function for pooled CL models expects subj num = (1-based) position in list of subjects with roi present (exclusing subjects without roi),
                # so need to correct indexing
                subj_num_adjusted = present_subjs.index(subj_num) + 1
                # print(subj_num_adjusted)
                ft_fmri, ft_nn = cl_model(data[0], data[1], subj_num_adjusted)
            else:
                ft_fmri, ft_nn = cl_model(data[0], data[1])
        # Flatten the features, collect them
        ft_fmri = ft_fmri.cpu().detach().numpy()
        ft_nn = ft_nn.cpu().detach().numpy()
        fmri_z[low_idx:high_idx] = ft_fmri
        nn_z[low_idx:high_idx] = ft_nn

    critic_out = (1 / tau) * cosine_similarity(nn_z, fmri_z)
    exp_critic = np.exp(critic_out)
    lower_bound_mi = 0
    for i in range(K):
        lower_bound_mi += np.log(exp_critic[i, i] /
                                 ((1 / K) * np.sum(exp_critic[i, :])))
    lower_bound_mi_unscaled = (1 / K) * lower_bound_mi

    # Do post-hoc scaling for temp
    betas = np.logspace(-2, 2, 1000)
    best_beta = 1
    best_lower_bound = lower_bound_mi_unscaled
    for beta in betas:
        tau_new = tau * beta
        critic_out = (1 / tau_new) * cosine_similarity(nn_z, fmri_z)
        exp_critic = np.exp(critic_out)
        lower_bound_mi = 0
        for i in range(K):
            lower_bound_mi += np.log(exp_critic[i, i] /
                                     ((1 / K) * np.sum(exp_critic[i, :])))
        lower_bound_mi = (1 / K) * lower_bound_mi
        if lower_bound_mi > best_lower_bound:
            best_lower_bound = lower_bound_mi
            best_beta = beta
    print(best_beta, best_lower_bound)

    # Save results
    if pooled:
        save_path = os.path.join(project_dir, "results", "Subj" + str(subj_num), "subj" + str(
            subj_num) + "_" + hemisphere_abbr + "h_mi_lower_bound_pooled_h" + h_method + ".joblib")
    else:
        save_path = os.path.join(project_dir, "results", "Subj" + str(subj_num), "subj" + str(subj_num) + "_" +
                                 hemisphere_abbr + "h_mi_lower_bound.joblib")
    try:
        results = joblib.load(save_path)
    except:
        results = {}
    results[roi] = lower_bound_mi_unscaled, np.log(
        K), best_beta, best_lower_bound
    joblib.dump(results, save_path)


# Load test cv results for single subject untuned for all layers and subjects to create figure 2 (matrix of encoding scores)
def load_test_cv_single_subj_untuned_all_layers(project_dir):
    num_rows = 46
    results_matrix = np.zeros((num_rows, 8))  # Average across subjects
    # Keep track of how many subjs have each ROI
    row_counters = np.zeros((num_rows))
    for subj_num in range(1, 9):
        results_folder_path = os.path.join(
            project_dir, "best_alex_out_layers_test_cv", "Subj" + str(subj_num))
        results_file = os.path.join(
            results_folder_path, "best_alex_layers_mat_untuned.npy")
        subj_results = np.load(results_file)
        for row_idx in range(num_rows):
            if subj_results[row_idx, 0] > 0:
                row_counters[row_idx] += 1
                results_matrix[row_idx, :] += subj_results[row_idx, :]
    for row_idx in range(num_rows):
        if results_matrix[row_idx, 0] > 0:
            results_matrix[row_idx, :] /= row_counters[row_idx]
    return results_matrix


# Load test cv results for single subject (untuned, cl-tuned, reg-tuned, pooled-avg, or pooled-specific)
# split options are all, early, higher
def load_test_cv_single_subj_results(project_dir, subj_num, split='all', return_pooled_results=False, roi=None):

    results_folder_path = os.path.join(
        project_dir, "best_alex_out_layers_test_cv", "Subj" + str(subj_num))
    untuned_results_path = os.path.join(
        results_folder_path, "best_alphas_voxel_accs_dict_untuned.joblib")
    cl_tuned_results_path = os.path.join(
        results_folder_path, "best_alphas_voxel_accs_dict_cl_tuned.joblib")
    reg_tuned_results_path = os.path.join(
        results_folder_path, "best_alphas_voxel_accs_dict_reg_tuned.joblib")
    pooled_avg_results_path = os.path.join(
        results_folder_path, "best_alphas_voxel_accs_dict_pooled_avg.joblib")
    pooled_const_results_path = os.path.join(
        results_folder_path, "best_alphas_voxel_accs_dict_pooled_const.joblib")

    untuned_results = joblib.load(untuned_results_path)
    cl_tuned_results = joblib.load(cl_tuned_results_path)
    reg_tuned_results = joblib.load(reg_tuned_results_path)
    pooled_avg_results = joblib.load(pooled_avg_results_path)
    pooled_const_results = joblib.load(pooled_const_results_path)

    if roi is not None:
        try:
            return untuned_results[roi], reg_tuned_results[roi], cl_tuned_results[roi], pooled_avg_results[roi], pooled_const_results[roi]
        except:
            return 0, 0, 0, 0, 0
    else:
        if split == 'all':
            rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
                    "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA",
                    "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
        elif split == 'early':
            rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
        elif split == 'higher':
            rois = ["EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA",
                    "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
        else:
            print("Unsupported Split!")
            return

        untuned_mean_acc = 0
        reg_tuned_mean_acc = 0
        cl_tuned_mean_acc = 0
        pooled_avg_mean_acc = 0
        pooled_const_mean_acc = 0
        cl_over_untuned_voxel_mean_percentage = 0
        cl_over_reg_tuned_voxel_mean_percentage = 0
        pooled_avg_over_cl_voxel_mean_percentage = 0
        pooled_const_over_cl_voxel_mean_percentage = 0
        num_rois_present = 0

        for hemi in ["lh", "rh"]:
            for roi in rois:
                try:
                    untuned_hemi_roi_results = untuned_results[hemi + '_' + roi]
                    reg_tuned_hemi_roi_results = reg_tuned_results[hemi + '_' + roi]
                    cl_tuned_hemi_roi_results = cl_tuned_results[hemi + '_' + roi]
                    pooled_avg_hemi_roi_results = pooled_avg_results[hemi + '_' + roi]
                    pooled_const_hemi_roi_results = pooled_const_results[hemi + '_' + roi]

                    untuned_mean_acc += untuned_hemi_roi_results.mean()
                    reg_tuned_mean_acc += reg_tuned_hemi_roi_results.mean()
                    cl_tuned_mean_acc += cl_tuned_hemi_roi_results.mean()
                    pooled_avg_mean_acc += pooled_avg_hemi_roi_results.mean()
                    pooled_const_mean_acc += pooled_const_hemi_roi_results.mean()

                    cl_over_untuned_voxel_mean_percentage += np.count_nonzero(
                        cl_tuned_hemi_roi_results - untuned_hemi_roi_results > 0) / cl_tuned_hemi_roi_results.shape[0]
                    cl_over_reg_tuned_voxel_mean_percentage += np.count_nonzero(
                        cl_tuned_hemi_roi_results - reg_tuned_hemi_roi_results > 0) / cl_tuned_hemi_roi_results.shape[0]

                    pooled_avg_over_cl_voxel_mean_percentage += np.count_nonzero(
                        pooled_avg_hemi_roi_results - cl_tuned_hemi_roi_results > 0) / cl_tuned_hemi_roi_results.shape[0]
                    pooled_const_over_cl_voxel_mean_percentage += np.count_nonzero(
                        pooled_const_hemi_roi_results - cl_tuned_hemi_roi_results > 0) / cl_tuned_hemi_roi_results.shape[0]

                    num_rois_present += 1
                except:
                    pass  # Skip missing ROIs

        untuned_mean_acc /= num_rois_present
        reg_tuned_mean_acc /= num_rois_present
        cl_tuned_mean_acc /= num_rois_present
        pooled_avg_mean_acc /= num_rois_present
        pooled_const_mean_acc /= num_rois_present

        cl_over_untuned_voxel_mean_percentage /= num_rois_present
        cl_over_reg_tuned_voxel_mean_percentage /= num_rois_present
        pooled_avg_over_cl_voxel_mean_percentage /= num_rois_present
        pooled_const_over_cl_voxel_mean_percentage /= num_rois_present

        if return_pooled_results:
            return pooled_avg_mean_acc, pooled_const_mean_acc, pooled_avg_over_cl_voxel_mean_percentage, pooled_const_over_cl_voxel_mean_percentage
        else:
            return untuned_mean_acc, reg_tuned_mean_acc, cl_tuned_mean_acc, cl_over_untuned_voxel_mean_percentage, cl_over_reg_tuned_voxel_mean_percentage


# Return matrix of results for each layer/roi
def load_test_cv_single_subj_results_all_layers(project_dir, subj_num):

    results_folder_path = os.path.join(
        project_dir, "best_alex_out_layers_test_cv", "Subj" + str(subj_num))
    untuned_results_path = os.path.join(
        results_folder_path, "best_alex_layers_mat_untuned.npy")
    cl_tuned_results_path = os.path.join(
        results_folder_path, "best_alex_layers_mat_cl_tuned.npy")
    reg_tuned_results_path = os.path.join(
        results_folder_path, "best_alex_layers_mat_reg_tuned.npy")
    pooled_avg_results_path = os.path.join(
        results_folder_path, "best_alex_layers_mat_pooled_avg.npy")
    pooled_const_results_path = os.path.join(
        results_folder_path, "best_alex_layers_mat_pooled_const.npy")

    untuned_results = np.load(untuned_results_path)
    cl_tuned_results = np.load(cl_tuned_results_path)
    reg_tuned_results = np.load(reg_tuned_results_path)
    pooled_avg_results = np.load(pooled_avg_results_path)
    pooled_const_results = np.load(pooled_const_results_path)

    return untuned_results, cl_tuned_results, reg_tuned_results, pooled_avg_results, pooled_const_results
