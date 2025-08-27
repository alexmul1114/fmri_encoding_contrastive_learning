import os
import time
import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio
import nibabel as nib
from scipy.stats import zscore
import nibabel as nib
import scipy.io as sio
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# Code adapted from: https://github.com/BNUCNL/NOD-fmri/blob/main/validation/nod_utils.py and
#                    https://github.com/BNUCNL/NOD-fmri/blob/main/validation/DNN-based_prf-encoding-model.py


# Dataset function to load corresponding (fmri response, image) pairs
class NODDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        support_path,
        device,
        subj_num,
        hemisphere,
        roi_name
    ):

        self.device = device

        # Get responses and image names
        self.fmri_roi, image_names = load_data(
            dataset_root, subj_num, roi_name, hemisphere, support_path)
        self.fmri_roi = torch.from_numpy(self.fmri_roi)

        # Find image paths
        self.img_paths = []
        stimuli_dir = os.path.join(dataset_root, "stimuli")
        for name in image_names:
            self.img_paths.append(os.path.join(stimuli_dir, name))

        # Define image transform
        self.transform = T.Compose([
            # resize the images to 224x24 pixels
            T.Resize((224, 224)),
            T.ToTensor(),  # convert the images to a PyTorch tensor
            # normalize the images color channels
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load fmri response
        fmri = self.fmri_roi[idx].to(self.device)
        # Load image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img).to(self.device)
        return fmri, img, idx


def get_NOD_dataloaders_cv(dataset_root, support_path, device, subj_num, hemisphere, roi_name, batch_size=1024, fold_num=0):

    # Seed generator
    torch.manual_seed(0)

    # Create dataset for fmri and image data
    dataset = NODDataset(dataset_root, support_path,
                         device, subj_num, hemisphere, roi_name)

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


# Returns brain response for selected subj, roi, hemi (top 50 voxels), and list
# of stimulus image paths
def load_data(dataset_root, subj_num, roi, hemisphere, support_path):

    clean_code = 'hp128_s4'
    ciftify_path = f'{dataset_root}/derivatives/ciftify'
    sub = "sub-0" + str(subj_num)
    if hemisphere == "left":
        hemi = "L"
    elif hemisphere == "right":
        hemi = "R"

    # Load full brain responses
    brain_resp = prepare_train_data(sub, support_path, code=clean_code)

    # Get top 50 voxels from selected ROI
    retino_path = f'{ciftify_path}/{sub}/results/ses-prf_task-prf'
    file_name = f'ses-prf_task-prf_params.dscalar.nii'
    retino_data = nib.load(pjoin(retino_path, file_name)).get_fdata()
    n_vertices = brain_resp.shape[-1]
    retinotopy_params = np.zeros((n_vertices, 3))
    retinotopy_params[:, 0] = retino_data[0, 0:n_vertices]
    retinotopy_params[:, 1] = retino_data[1, 0:n_vertices]*16/200
    retinotopy_params[:, 2] = retino_data[2, 0:n_vertices]*16/200
    # R2
    r2 = retino_data[3, 0:59412]
    roi_indices = np.where(get_roi_data(None, roi, support_path, hemi) == 1)
    roi_r2 = r2[roi_indices]
    if np.isnan(roi_r2).sum():
        print(f'{roi} with value NaN')
        roi_r2[np.where(np.isnan(roi_r2) == 1)] = 0
    top50 = np.argsort(roi_r2)[-50::]
    voxel_indices = roi_indices[0][top50]
    roi_hemi_brain_resp = brain_resp[:, voxel_indices]

    # Get stimulus image paths
    stim_paths = []
    imagenet_label_path = os.path.join(
        support_path, f'{sub}_imagenet-label.csv')
    with open(imagenet_label_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stim_paths.append(row["image_name"].strip())

    return roi_hemi_brain_resp, stim_paths


# sub_names format: "sub-01"
def prepare_imagenet_data(dataset_root, sub_names=None, support_path=os.path.join("/Users", "alexm", "fmri_cl_data", "nod", "supportfiles"), clean_code='hp128_s4'):
    # define path
    ciftify_path = f'{dataset_root}/derivatives/ciftify'
    nifti_path = f'{dataset_root}'
    if not sub_names:
        sub_names = sorted([i for i in os.listdir(
            ciftify_path) if i.startswith('sub')])
    n_class = 1000
    num_ses, num_run, num_trial = 4, 10, 100
    vox_num = 59412
    # for each subject
    for sub_idx, sub_name in enumerate(sub_names):
        print(sub_name)
        label_file = pjoin(support_path, f'{sub_name}_imagenet-label.csv')
        # check whether label exists, if not then generate
        if not os.path.exists(label_file):
            sub_events_path = pjoin(nifti_path, sub_name)
            df_img_name = []
            # find imagenet task
            imagenet_sess = [_ for _ in os.listdir(sub_events_path) if (
                'imagenet' in _) and ('05' not in _)]
            imagenet_sess.sort()  # Remember to sort list !!!
            # loop sess and run
            for sess in imagenet_sess:
                for run in np.linspace(1, 10, 10, dtype=int):
                    # open ev file
                    events_file = pjoin(sub_events_path, sess, 'func',
                                        '{:s}_{:s}_task-imagenet_run-{:02d}_events.tsv'.format(sub_name, sess, run))
                    tmp_df = pd.read_csv(events_file, sep="\t")
                    df_img_name.append(
                        tmp_df.loc[:, ['trial_type', 'stim_file']])
            df_img_name = pd.concat(df_img_name)
            df_img_name.columns = ['class_id', 'image_name']
            df_img_name.reset_index(drop=True, inplace=True)
            # add super class id
            superclass_mapping = pd.read_csv(
                pjoin(support_path, 'superClassMapping.csv'))
            superclass_id = superclass_mapping['superClassID'].to_numpy()
            class_id = (
                df_img_name.loc[:, 'class_id'].to_numpy()-1).astype(int)
            df_img_name = pd.concat([df_img_name, pd.DataFrame(
                superclass_id[class_id], columns=['superclass_id'])], axis=1)
            # make path
            if not os.path.exists(support_path):
                os.makedirs(support_path)
            df_img_name.to_csv(label_file, index=False)
            print(f'Finish preparing labels for {sub_name}')
        # load sub label file
        label_sub = pd.read_csv(label_file)['class_id'].to_numpy()
        label_sub = label_sub.reshape((num_ses, n_class))
        # define beta path
        beta_sub_path = pjoin(
            support_path, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
        if not os.path.exists(beta_sub_path):
            # extract from dscalar.nii
            beta_sub = np.zeros((num_ses, num_run*num_trial, vox_num))
            for i_ses in range(num_ses):
                for i_run in range(num_run):
                    run_name = f'ses-imagenet{i_ses+1:02d}_task-imagenet_run-{i_run+1}'
                    beta_data_path = pjoin(
                        ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                    beta_sub[i_ses, i_run*num_trial: (i_run + 1)*num_trial, :] = np.asarray(
                        nib.load(beta_data_path).get_fdata())
            # save session beta in ./supportfiles
            np.save(beta_sub_path, beta_sub)


def train_data_normalization(data, metric='run', runperses=10, trlperrun=100):
    if data.ndim != 2:
        raise AssertionError(
            'check data shape into (n-total-trail,n-brain-size)')
        return 0
    if metric == 'run':
        nrun = data.shape[0] / trlperrun
        for i in range(int(nrun)):
            # run normalization is to demean the run effect
            data[i*trlperrun:(i+1)*trlperrun, :] = zscore(data[i *
                                                               trlperrun:(i+1)*trlperrun, :], None)
    elif metric == 'session':
        nrun = data.shape[0] / trlperrun
        nses = nrun/runperses
        for i in range(int(nses)):
            data[i*trlperrun*runperses:(i+1)*trlperrun*runperses, :] = zscore(
                data[i*trlperrun*runperses:(i+1)*trlperrun*runperses, :], None)
    elif metric == 'trial':
        data = zscore(data, axis=1)
    return data


def prepare_train_data(sub, support_path, code='hp128_s4', metric='run', runperses=10, trlperrun=100):
    file_name = f'{sub}_imagenet-beta_{code}_ridge.npy'
    brain_resp = np.load(pjoin(support_path, file_name))
    brain_resp = brain_resp.reshape((-1, brain_resp.shape[-1]))
    brain_resp = train_data_normalization(
        brain_resp, metric, runperses, trlperrun)
    return brain_resp


def get_voxel_roi(voxel_indices, support_path):
    roi_info = pd.read_csv(os.path.join(support_path, 'roilbl_mmp.csv'), sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:, 0].values))
    roi_brain = sio.loadmat(
        os.path.join(support_path,'MMP_mpmLR32k.mat'))['glasser_MMP'].reshape(-1)
    if roi_brain[voxel_indices] > 180:
        return roi_list[int(roi_brain[voxel_indices]-181)]
    else:
        return roi_list[int(roi_brain[voxel_indices]-1)]


def get_roi_data(data, roi_name, support_path, hemi=False):
    roi_info = pd.read_csv(os.path.join(
        support_path, 'roilbl_mmp.csv'), sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:, 0].values))
    roi_brain = sio.loadmat(
        os.path.join(support_path, 'MMP_mpmLR32k.mat'))['glasser_MMP'].reshape(-1)
    if data is not None:
        if data.shape[1] == roi_brain.size:
            if not hemi:
                return np.hstack((data[:, roi_brain == (1+roi_list.index(roi_name))], data[:, roi_brain == (181+roi_list.index(roi_name))]))
            elif hemi == 'L':
                return data[:, roi_brain == (1+roi_list.index(roi_name))]
            elif hemi == 'R':
                return data[:, roi_brain == (181+roi_list.index(roi_name))]
        else:
            roi_brain = np.pad(
                roi_brain, (0, data.shape[1]-roi_brain.size), 'constant')
            if not hemi:
                return np.hstack((data[:, roi_brain == (1+roi_list.index(roi_name))], data[:, roi_brain == (181+roi_list.index(roi_name))]))
            elif hemi == 'L':
                return data[:, roi_brain == (1+roi_list.index(roi_name))]
            elif hemi == 'R':
                return data[:, roi_brain == (181+roi_list.index(roi_name))]
    else:
        roi_brain = np.pad(roi_brain, (0, 91282-roi_brain.size), 'constant')
        if type(roi_name) == list:
            return np.sum([get_roi_data(None, _, support_path, hemi) for _ in roi_name], axis=0)
        else:
            if not hemi:
                return (roi_brain == (1+roi_list.index(roi_name))) + (roi_brain == (181+roi_list.index(roi_name)))
            elif hemi == 'L':
                return roi_brain == (1+roi_list.index(roi_name))
            elif hemi == 'R':
                return roi_brain == (181+roi_list.index(roi_name))


# def make_stimcsv(sub, data_path, support_path):
#     save_path = os.path.join(support_path, "sub_stim")
#     if not os.path.exists(save_path):
#             os.makedirs(save_path)
#     header = ['type=image\n',f'path={data_path}/stimuli/\n',
#                 f'title=ImageNet images in {sub}\n','data=stimID\n']
#     # stim files
#     sub_stim = pd.read_csv(pjoin(support_path, f'{sub}_imagenet-label.csv'), sep=',')
#     # replace file name
#     stim_files = '\n'.join(sub_stim['image_name'].tolist())
#     with open(f'{save_path}/{sub}_imagenet.stim.csv', 'w') as f:
#         f.writelines(header)
#         f.writelines(stim_files)

# # Create stimulus image paths csv
# def prepare_imagenet_paths(sub, data_path, support_path):
#     stim_path = os.path.join(support_path, "sub_stim")
#     if not os.path.exists(stim_path):
#         os.makedirs(stim_path)
#     stimcsv = pjoin(stim_path, f'{sub}_imagenet.stim.csv')
#     if not os.path.exists(stimcsv):
#         make_stimcsv(sub, data_path, support_path)
