# fmri_encoding_contrastive_learning
Code and results files for fmri encoding with contrastive learning paper.

## Results
The Results directory contains subject-specific results for all subjects and ROIs (untuned, regression-tuned, and CL-tuned methods), raw cross-subject results  as npy files, pictures for cross subject results, raw results for each Subject and ROI for the image classification tasks, and the pictures for the image classification results.

## Code
**algonauts_models.py** - defines classes and functions for CL models and regression models <br>
**algonauts_model_training.py** - script that can be run from the command line to train CL or regression models. Options are specified at the beginning of the file. <br>
**algonauts_utils.py** - contains helper functions for dataset loading, mapping voxels to correct ROIs, etc. <br>


