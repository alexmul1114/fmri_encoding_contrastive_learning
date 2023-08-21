# Imports

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import torch

from argparse import ArgumentParser

# Local imports
from utils import get_dataloaders, get_dataloaders_unshuffled
from models import get_CL_model, train


# Command line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to main folder of project", required=True)
parser.add_argument("--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument("--subj", help="Subject number, 1 through 8", type=int, required=True)
parser.add_argument("--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--batch-size", help="Batch size for training", type=int, default=1024)
parser.add_argument("--epochs", help="Number of epochs to train model", type=int, default=30)



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
    
    all_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1",
                "FBA-2", "mTL-bodies", "OFA", "FFA-1",
                   "FFA-2", "mTL-faces", "aTL-faces", "OPA",
            "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early",
            "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]

    save_dir = project_dir + r"/cl_models/Subj" + str(subj_num) 
    os.chdir(save_dir)
    
    # Train a model for each roi
    for roi in all_rois:
        
        # Get dataloaders
        train_dataloader, test_dataloader, train_size, test_size, num_voxels = get_dataloaders_unshuffled(project_dir, device, subj_num, hemisphere, roi, batch_size)
        # Handle empty ROIs
        #print(num_voxels)
        if (num_voxels==0):
            print(roi + " is empty")
        elif (num_voxels > 4000):
            print("ROI is too large to train model")
        else:
            # Get model, optimizer, and scheduler
            model, optimizer, scheduler = get_CL_model(num_voxels, device)

            # Train model
            print("Training model for " + roi + ":")
            trained_model = train(model, device, train_dataloader, test_dataloader, optimizer, scheduler, epochs, temp=0.3)

            # Save model in models directory
            hemisphere_abbr = 'l' if hemisphere=='left' else 'r'
            model_name = "subj" + str(subj_num) + "_" + hemisphere_abbr + "h_" + roi + "_model_e" + str(epochs) + ".pt"
            torch.save(trained_model, model_name)
            
            del model, optimizer, scheduler, trained_model
    
    



