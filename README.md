# AnatGuidedCMBSeg
Official implementation of **Anatomically-Guided Segmentation of Cerebral Microbleeds in T1-weighted and T2\*-weighted MRI**.

## News
- (Jun. 2024) Our paper has been accepted to MICCAI 2024!

## Installation
1. Install Python 3.8 and PyTorch 1.11.0. We recommend to use [**pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime**](https://hub.docker.com/layers/pytorch/pytorch/1.11.0-cuda11.3-cudnn8-runtime/images/sha256-9904a7e081eaca29e3ee46afac87f2879676dd3bf7b5e9b8450454d84e074ef0)
2. Install [**nnUNet-v1**](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)
3. Install [**SimpleITK**](https://simpleitk.org/)
4. `git clone https://github.com/junmokwon/AnatGuidedCMBSeg`
5. Install [**ANTs**](https://github.com/ANTsX/ANTs) and [**FreeSurfer**](https://surfer.nmr.mgh.harvard.edu/) including [**SynthStrip**](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) for MRI preprocessing.

## T1-weighted MRI Preprocessing
1. Ensure that T1 and T2* MRI scans have the same orientation, resolution, and fields of view.
2. Ensure there are no oblique or orientation issues in T1-weighted MRI data.
3. Run Freesurfer `recon-all`

## T2*-weighted MRI Preprocessing
1. Ensure that T1 and T2* MRI scans have the same orientation, resolution, and fields of view.
2. Ensure there are no oblique or orientation issues in T2*-weighted MRI data.
3. Run SynthStrip skull stripping `mri_synthstrip`
4. Run N4ITK `N4BiasFieldCorrection`
5. Perform rigid-body registration from T1 space to T2* space.

## JHU-DTI Atlas Preprocessing
1. Download JHU-DTI Atlas
2. Prepare white-matter atlases including internal capsule and external capsule.
3. Perform rigid-body registration `antsRegistrationSyN.sh -t r` from MNI152 space to MNI305 (Talairach) space.
4. Transform IC and EC labels to Talairach space.

## Proxy Label Generation
1. Generate lobar parcellation to obtain `aparc.lobes.mgz` and `wmparc.lobes.mgz`
2. Merge cerebral lobes from `aparc.lobes.mgz` and `wmparc.lobes.mgz` into lobar region.
3. Transform internal capsule and external capsule labels from MNI305 (Talairach) space to subject’s native T1 space using Talairach transform `transforms/talairach.xfm`
4. Merge deep white matter regions from `aparc.lobes.mgz`, `wmparc.lobes.mgz`, and JHU-DTI atlas into deep supratentorial region.
5. Merge brainstem and cerebellum from `aparc.lobes.mgz` into infratentorial region.

## nn-UNet Training
1. Preprocess T1 and T2* MRI scans.
2. Generate target labels: lobar, deep supratentorial, infratentorial, and CMB labels.
3. Choose a task ID e.g., `Task301_InHouse` and `Task302_VALDO2021`
4. Run `nnUNet_plan_and_preprocess -t 301` where 301 is Task ID.
5. Train a 3D full resolution nn-UNet with `nnUNetTrainerV2_Loss_DiceTopK10` trainer.

## Clinically-derived False Positive Reduction
1. Run `nnUNet_predict` to predict hard-thresholded segmentation masks for proxy and CMB labels.
2. For each connected component in CMB prediction masks, do the following:
3. Generate peripheral mask by performing logical XOR between dilated mask and predicted mask.
4. Calculate brain parenchyma ratio in peripheral mask using proxy prediction masks.
5. Discard the connected component if the ratio does not exceed 0.5
