# fmri_encoding_contrastive_learning
Code and results files for fmri encoding with contrastive learning paper.

## Directions
1. Download the images, fMRI data, and ROI masks from the Algonauts Project website: [http://algonauts.csail.mit.edu/challenge.html] (http://algonauts.csail.mit.edu/challenge.html). Put the fMRI data in a subfolder called training_fmri in the root project directory, and the images in a subfolder called training_images. Both of those folders should have one folder for each subject named Subjx. For the fMRI data, each subject folder should have two files: subjx_lh_training_fmri.npy and subjx_rh_training_fmri.npy. Each subject folder in the training_images directory should have all of the images for that subject (with names like train-0001_nsd-00013, as downloaded). The ROI masks should be in a roi_masks folder in the root project directory, also with one folder for each subject called Subjx that contains all of the mask files (lh.all-vertices_fsaverage_space, rh.floc-faces_challenge_space, etc.)
2. Before doing anything else, run the generate_voxel_roi_mappings.py script, which creates a folder called voxel_roi_mappings in the root directory, with a folder for each subject containing files that map all the voxels in the left and right hemisphere to the ROIs they are in.
3. Train CL, regression, or pooled models using the model training scripts. 
4. Once models are trained, they can be used to get encoding results with the **fit_encoding_models** script.

## Code
For all runable scripts, options are specified at the begining of the file. <br>
**generate_voxel_roi_mappings.py** -- script that should be **run first** to create mappings from voxels to the ROIs they are in for all voxels in each subject <br>
**utils.py** - contains helper functions for dataset loading, mapping voxels to correct ROIs, extracting features from AlexNet models, etc. <br>
**results_utils.py** - contains helper functions for downstream tasks, mutual information estimates, more. <br>
**models.py** - defines classes and functions for CL models and regression models <br>
**cl_model_training.py, cl_model_training_all_rois.py, reg_model_training.py, reg_model_training_all_rois.py, pooled_model_training.py** - scripts that can be run from the command line to train CL or regression models for each a single ROI (model_training files) or all the ROIs in a hemisphere for the specified subject (all_rois files), or for the pooled approach (using all subjects' data). <br>
**fit_encoding_models.py** - script that used untuned, cl-tuned, reg-tuned, or pooled-tuned AlexNet as feature extractor to fit encoding models <br>
**generate_results.py** - script that can be used to generate results for downstream classification tasks, mutual information lower bound estimates, more.


