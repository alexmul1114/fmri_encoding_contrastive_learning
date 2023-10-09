# Imports

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import torch

from argparse import ArgumentParser

# Local imports
from utils import get_dataloaders
from models import get_CL_model, train, get_reg_model, train_reg


# Command line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to main folder of project", required=True)
parser.add_argument("--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument("--subj", help="Subject number, 1 through 8", type=int, required=True)
parser.add_argument("--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--roi", help="ROI", type=str, required=True)
parser.add_argument("--batch-size", help="Batch size for training", type=int, default=1024)
parser.add_argument("--epochs", help="Number of epochs to train model", type=int, default=75)



if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Load arguments
    project_dir = args.root
    device = args.device
    device = torch.device(device)
    subj_num = args.subj
    hemisphere = args.hemisphere
    if (hemisphere != 'left' and hemisphere != 'right'):
        print("Invalid hemisphere, must be left or right")
        sys.exit()
    batch_size = args.batch_size
    epochs = args.epochs
    roi = args.roi


    save_dir = project_dir + r"/baseline_models/nn_reg/Subj" + str(subj_num) 
    os.chdir(save_dir)
    
    # Train a model for each roi
        
    # Get dataloaders
    train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders(project_dir, device, subj_num, hemisphere, roi, batch_size)
    # Handle empty ROIs
    if (num_voxels==0):
        print(roi + " is empty")
    elif (num_voxels > 4500):
        print("ROI is too large to train model")
    else:
        # Get model, optimizer, and scheduler
        model, optimizer = get_reg_model(num_voxels, device, lr=0.000025)

        # Train model
        print("Training reg model for subject " + str(subj_num) + " " + str(hemisphere) + " hemisphere - " + roi)
        #print("Training regression model for " + roi + ":")
        trained_model = train_reg(model, device, train_dataloader, optimizer, epochs)

        # Save model in models directory
        hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
        model_name = "subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_reg_model_e" + str(epochs) + ".pt"
        torch.save(trained_model.state_dict(), model_name)