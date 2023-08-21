# Imports
from argparse import ArgumentParser

import torch

import sys

# Local imports
from utils import get_dataloaders, get_dataloaders_unshuffled, fit_pca, extract_pca_features, get_fmri_from_dataloader, get_dataloaders_with_img_paths
from models import CLR_model, fmri_reg
from results_utils import tsne_visualization, get_violin_plot, get_results_cross_subj, get_results_single_subj_all_rois, get_people_classifier_results, image_classification_results



# Command line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to main folder of project", required=True)
parser.add_argument("--device", help="Device to use for training, cpu or cuda", required=True)
parser.add_argument("--subj", help="Subject number, 1 through 8", type=int, required=True)
parser.add_argument("--hemisphere", help="Hemisphere, either left or right", type=str, required=True)
parser.add_argument("--type", help="Options are single-subj, cross-subj, violin, classification-task", type=str, required=True)
parser.add_argument("--comparator", help="comparison method for violin plot (Control or Regression)", type=str, required=False, default="Control")
parser.add_argument("--rois", help="List of rois", nargs='+', type=str, required=False, default=[])
parser.add_argument("--training", help="Whether to use training results for cross-subj", type=bool, required=False, default=False)
parser.add_argument("--dataset", help="Image dataset to use for classification task", type=str, required=False, default='caltech256')
parser.add_argument("--tuningmethod", help="Tuning method to use for classification task, cl or reg", type=str, required=False, default='cl')

    
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Load arguments
    project_dir = args.root
    device = args.device
    device = torch.device(device)
    subj_num = args.subj
    hemisphere = args.hemisphere
    method = args.type
    rois = args.rois
    comparator = args.comparator
    use_training_results = args.training
    img_dataset = args.dataset
    tuning_method = args.tuningmethod


    if (hemisphere != 'left' and hemisphere != 'right'):
        print("Invalid hemisphere, must be left or right")
        sys.exit()
        
    if (method=='single-subj'):
        get_results_single_subj_all_rois(project_dir, device, subj_num, hemisphere)
    elif (method=='cross-subj'):
        for roi in rois:
            get_results_cross_subj(project_dir, device, hemisphere, roi, num_subjs=8, reg=True, training_results=use_training_results)
    elif (method=="violin"):
        get_violin_plot(subj_num, hemisphere, rois, comparator)
    elif (method=='classification-task'):
        image_classification_results(project_dir, subj_num, hemisphere, rois, device, tuning_method, img_dataset, save=True)
    else:
        print("Invalid method")
        
