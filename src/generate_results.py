
from argparse import ArgumentParser
import torch
import sys
from utils import get_n_random_rois
from results_utils import image_classification_results, compute_mi_lower_bound
from utils import make_voxels_counts_file


# Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--root", help="Path to main folder of project", required=True)
parser.add_argument("--subj", help="Subject number, 1 through 8",
                    type=int, required=False, default=1)
parser.add_argument("--hemisphere", help="Hemisphere, either left or right",
                    type=str, required=False, default="left")
parser.add_argument("--type", type=str, required=True)
parser.add_argument("--rois", help="List of rois", nargs='+',
                    type=str, required=False, default=[])
parser.add_argument("--dataset", help="Image dataset to use for classification task",
                    type=str, required=False, default='caltech256')
parser.add_argument("--tuning-method", help="Tuning method to use for classification task, cl or reg",
                    type=str, required=False, default='cl')


if __name__ == '__main__':

    args = parser.parse_args()

    # Load arguments
    project_dir = args.root
    subj_num = args.subj
    hemisphere = args.hemisphere
    method = args.type
    rois = args.rois
    use_training_results = args.training
    img_dataset = args.dataset
    tuning_method = args.tuning_method

    device = torch.device('cpu')

    if hemisphere != 'left' and hemisphere != 'right':
        print("Invalid hemisphere, must be left or right")
        sys.exit()

    if method == 'random-rois-list':
        get_n_random_rois(project_dir, n=30)
    elif method == 'classification-task':
        image_classification_results(project_dir, subj_num, hemisphere, rois,
                                     device, tuning_method, img_dataset, save=True, save_probs=True)
    elif method == 'classification-task-pooled-avg':
        image_classification_results(project_dir, subj_num, hemisphere, rois, device, tuning_method,
                                     img_dataset, save=True, pooled=True, pooled_h_method='avg', save_probs=True)
    elif method == 'classification-task-pooled-const':
        image_classification_results(project_dir, subj_num, hemisphere, rois, device, tuning_method,
                                     img_dataset, save=True, pooled=True, pooled_h_method='const', save_probs=True)
    elif method == "lower-bound-mi":
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
                "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "OPA",
                "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
        for roi in rois:
            compute_mi_lower_bound(project_dir, device,
                                   subj_num, roi, hemisphere)
    elif method == "lower-bound-mi-pooled-const":
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
                "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "OPA",
                "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
        for roi in rois:
            compute_mi_lower_bound(
                project_dir, device, subj_num, roi, hemisphere, pooled=True, h_method='const')
    elif method == "lower-bound-mi-pooled-avg":
        rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2",
                "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "OPA",
                "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]
        for roi in rois:
            compute_mi_lower_bound(
                project_dir, device, subj_num, roi, hemisphere, pooled=True, h_method='avg')
    elif method == "voxel-counts":
        make_voxels_counts_file(project_dir, hemisphere)
    else:
        print("Invalid method")
