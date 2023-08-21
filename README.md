# fmri_encoding_contrastive_learning
Code and results files for fmri encoding with contrastive learning paper.

## Results
The Results directory contains subject-specific results for all subjects and ROIs (untuned, regression-tuned, and CL-tuned methods), raw cross-subject results  as npy files, pictures for cross subject results, raw results for each Subject and ROI for the image classification tasks, and the pictures for the image classification results.

## Code
For all runable scripts, options are specified at the begining of the file.
**models.py** - defines classes and functions for CL models and regression models <br>
**utils.py** - contains helper functions for dataset loading, mapping voxels to correct ROIs, etc. <br>
**results_utils.py** -- helper functions for generating single-subject, cross-subject, and image classification results <br>
**generate_voxel_roi_mappings.py** -- script that should be run to create mappings from voxels to the ROIs they are in for all voxels in each subject <br>
**get_best_alex_layer.py** -- script that can be run to determine the untuned AlexNet layer which gives the highest encoding accuracy after a linear mapping. <br>
**cl_model_training.py, cl_model_training_all_rois.py, reg_model_training.py, reg_model_training_all_rois.py** - scripts that can be run from the command line to train CL or regression models for each a single ROI (model_training files) or all the ROIs in a hemisphere for the specified subject (all_rois files). <br>
**generate_results.py** - scripts that can be run from the command line to generate results (single-subject, cross-subject, or image classification tasks) once models are trained

## Directions
1. Download the images, fMRI data, and ROI masks from the Algonauts Project website: [http://algonauts.csail.mit.edu/challenge.html] (http://algonauts.csail.mit.edu/challenge.html). Put the fMRI data in a subfolder called training_fmri in the root project directory, and the images in a subfolder called training_images. Both of those folders should have one folder for each subject name Subjx. For the fMRI data, each subject folder should have two files: subjx_lh_training_fmri.npy and subjx_rh_training_fmri.npy. Each subject folder in the training_images directory should have all of the images for that subject (with names like train-0001_nsd-00013, as downloaded). The ROI masks should be in a roi_masks folder in the root project directory, also with one folder for each subject called Subjx that contains all of the mask files (lh.all-vertices_fsaverage_space, rh.floc-faces_challenge_space, etc.)
2. Before doing anything else, run the generate_voxel_roi_mappings.py script, which creates a folder called voxel_roi_mappings in the root directory, with a folder for each subject containing files that map all the voxels in the left and right hemisphere to the ROIs they are in.
3. 



