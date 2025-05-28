
from models import get_pooled_CL_model, train_pooled
from utils import get_dataloaders
from argparse import ArgumentParser
import numpy as np
import torch
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


# Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--root", help="Path to main folder of project", required=True)
parser.add_argument(
    "--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument(
    "--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--roi", help="ROI", type=str, required=True)
parser.add_argument(
    "--h-method", help="Method for setting dimension of h, options are constant, average", type=str, default="constant")
parser.add_argument(
    "--batch-size", help="Batch size for training", type=int, default=1024)
parser.add_argument(
    "--epochs", help="Number of epochs to train model", type=int, default=30)


if __name__ == '__main__':

    args = parser.parse_args()

    # Load arguments
    project_dir = args.root
    device = args.device
    device = torch.device(device)
    hemisphere = args.hemisphere
    h_method = args.h_method
    if hemisphere != 'left' and hemisphere != 'right':
        print("Invalid hemisphere, must be left or right")
        sys.exit()
    roi = args.roi
    batch_size = args.batch_size
    epochs = args.epochs

    # Get dataloaders
    train_dataloaders_subjs = []
    num_voxels_subjs = []
    for subj_num in range(1, 9):
        train_dataloader, _, _, _, num_voxels = get_dataloaders(
            project_dir, device, subj_num, hemisphere, roi, batch_size)
        if num_voxels != 0:
            train_dataloaders_subjs.append(train_dataloader)
            num_voxels_subjs.append(num_voxels)
    print(hemisphere, roi)
    print("Number of subjects with ROI present: " + str(len(num_voxels_subjs)))
    print("Number of voxels for each subject:", num_voxels_subjs)

    # Determine dimension of h based on selected method
    if h_method.lower() == "constant":
        h_dim = 5741
    elif h_method.lower() == "average":
        h_dim = int(np.mean(np.array(num_voxels_subjs)))
        print("Average number of voxels (h-dim): " + str(h_dim))

    # Get model, optimizer, and scheduler
    model, optimizer = get_pooled_CL_model(
        num_voxels_subjs, device, h_dim, lr=1e-4)

    # Train model
    print("Training pooled cl model for " +
          str(hemisphere) + " hemisphere - " + roi)
    trained_model = train_pooled(
        model, device, train_dataloaders_subjs, optimizer, epochs, temp=0.3)

    # Save model in models directory
    save_dir = os.path.join(project_dir, "cl_models")
    hemisphere_abbr = 'l' if hemisphere == 'left' else 'r'
    if h_method.lower() == 'constant':
        model_name = hemisphere_abbr + "h_" + roi + \
            "_pooled_model_e" + str(epochs) + "_hconst.pt"
    elif h_method.lower() == 'average':
        model_name = hemisphere_abbr + "h_" + roi + \
            "_pooled_model_e" + str(epochs) + "_havg.pt"

    os.chdir(save_dir)
    torch.save(trained_model.state_dict(), model_name)
