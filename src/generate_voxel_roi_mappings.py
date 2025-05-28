
import os
import sys
import joblib
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from utils import get_voxel_roi_mappings, get_dataloaders


# Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--root", help="Path to main folder of project", required=True)


if __name__ == '__main__':

    # Load argument
    args = parser.parse_args()
    project_dir = args.root

    device = 'cpu'
    device = torch.device(device)

    # Go over all subjects
    print("Finding voxel-ROI mappings for each subject and hemisphere...")
    for subj_num in tqdm(range(1, 9)):

        save_folder = os.path.join(
            project_dir, "voxel_roi_mappings", "Subj" + str(subj_num))
        os.chdir(save_folder)

        # Create dataloader for left hemisphere
        train_dataloader, train_size, num_voxels = get_dataloaders(project_dir, device, subj_num, 'left', 'all',
                                                                   batch_size=1024, voxel_subset=False, voxel_subset_num=0, use_all_data=True)
        # Get list of voxel-ROI mappings
        voxel_roi_lists = get_voxel_roi_mappings(
            project_dir, subj_num, 'left', num_voxels)

        file_name = "subj" + str(subj_num) + "_lh_voxel_roi_mappings.joblib"
        joblib.dump(voxel_roi_lists, file_name)

        # Create dataloader for right hemisphere
        train_dataloader, train_size, num_voxels = get_dataloaders(project_dir, device, subj_num, 'right', 'all',
                                                                   batch_size=1024, voxel_subset=False, voxel_subset_num=0, use_all_data=True)
        # Get list of voxel-ROI mappings
        voxel_roi_lists = get_voxel_roi_mappings(
            project_dir, subj_num, 'right', num_voxels)

        file_name = "subj" + str(subj_num) + "_rh_voxel_roi_mappings.joblib"
        joblib.dump(voxel_roi_lists, file_name)
