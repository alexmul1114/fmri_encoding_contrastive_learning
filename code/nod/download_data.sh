#!/bin/bash

mkdir -p data
cd data
datalad clone ///openneuro/ds004496 ds004496
cd ds004496

# Get beta files
for subj in {01..09}; do
  for ses in {01..05}; do
    for run in {1..10}; do
      datalad get -r \
        "derivatives/ciftify/sub-${subj}/results/ses-imagenet${ses}_task-imagenet_run-${run}"
    done
  done
done

# Get retino data files
for subj in {01..09}; do
  datalad get -r \
    "derivatives/ciftify/sub-${subj}/results/ses-prf_task-prf/ses-prf_task-prf_params.dscalar.nii"
done

# Get all the label.txt files
git ls-files \
  | grep -E '^derivatives/ciftify/sub-0[1-9]/results/.*ses-imagenet0[1-5]_task-imagenet_run-[0-9]{2}_label\.txt$' \
  | while IFS= read -r file; do
      git annex get "$file"
    done

# Get imagenet stimuli
datalad get stimuli/imagenet/**

cd ..
cd ..
mkdir -p supportfiles

# Get support files
curl -L -o nod-fmri.zip \
  https://github.com/BNUCNL/NOD-fmri/archive/refs/heads/main.zip
unzip nod-fmri.zip 'NOD-fmri-main/validation/supportfiles/*' -d .
mv NOD-fmri-main/validation/supportfiles supportfiles
rm -rf NOD-fmri-main nod-fmri.zip