# fmri_encoding_contrastive_learning
Code and results files for fmri encoding with contrastive learning paper.

## Results
The Results directory contains subject-specific results for all subjects and ROIs (untuned, regression-tuned, and CL-tuned methods), raw cross-subject results  as npy files, pictures for cross subject results, raw results for each Subject and ROI for the image classification tasks, and the pictures for the image classification results.

## Code
For all runable scripts, options are specified at the begining of the file.
**models.py** - defines classes and functions for CL models and regression models <br>
**utils.py** - contains helper functions for dataset loading, mapping voxels to correct ROIs, etc. <br>
**results_utils.py** -- helper functions for generating single-subject, cross-subject, and image classification results <br>
**generate_voxel_roi_mappings.py** -- <br>
**get_best_alex_layer.py** -- <br>
**cl_model_training.py, cl_model_training_all_rois.py, reg_model_training.py, reg_model_training_all_rois.py** - scripts that can be run from the command line to train CL or regression models for each a single ROI (model_training files) or all the ROIs in a hemisphere for the specified subject (all_rois files). <br>
**generate_results.py** - scripts that can be run from the command line to generate results (single-subject, cross-subject, or image classification tasks) once models are trained



