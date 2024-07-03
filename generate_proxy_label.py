import os
from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk


LOBAR_INDEX = 1
DEEP_INDEX = 2
INFRA_INDEX = 3


def convert_mgz_to_nii(src):
    src = Path(src)
    out = (src.parent / src.name).with_suffix(".nii.gz")
    if not out.exists():
        os.system(f'mri_convert "{src}" "{out}"')


def load_array(src):
    img = sitk.ReadImage(str(src))
    array = sitk.GetArrayFromImage(img)
    return array


def generate_target_label(args):
    # Sanity check for arguments
    sd = Path(args.sd)
    assert sd.exists(), f'Input directory does not exist: "{sd}"'
    case = args.subject
    mri_dir = sd / case / 'mri'
    assert mri_dir.exists(), f'Recon-all not performed: "{mri_dir}"'
    jhu_mni152 = Path(args.jhu_dti_atlas)
    assert jhu_mni152.exists(), f'JHU-DTI atlas not found: "{jhu_mni152}"'

    # Sanity check for recon-all
    aparc_aseg = mri_dir / 'aparc+aseg.mgz'
    assert aparc_aseg.exists(), f'aparc+aseg.mgz not found: "{aparc_aseg}"'
    wmparc = mri_dir / 'wmparc.mgz'
    assert wmparc.exists(), f'wmparc.mgz not found: "{wmparc}"'
    brain = mri_dir / 'brain.mgz'
    assert brain.exists(), f'brain.mgz not found: "{brain}"'

    # Lobar parcellation from FreeSurferColorLUT.txt
    aparc_lobes = mri_dir / 'aparc.lobes.mgz'
    if not aparc_lobes.exists():
        commands = [
            f'export SUBJECTS_DIR={sd}',
            f'mri_annotation2label --sd {sd} --subject {case} --hemi lh --lobesStrict lobes',
            f'mri_annotation2label --sd {sd} --subject {case} --hemi rh --lobesStrict lobes',
            f'mri_aparc2aseg --s {case} --rip-unknown --volmask --o "{aparc_lobes}" --annot lobes --base-offset 300',
        ]
        os.system('\n'.join(commands))
        assert aparc_lobes.exists(), f'aparc.lobes.mgz not found: "{aparc_lobes}"'
    
    # Lobar white matter parcellation from FreeSurferColorLUT.txt
    wmparc_lobes = mri_dir / 'wmparc.lobes.mgz'
    if not wmparc_lobes.exists():
        commands = [
            f'export SUBJECTS_DIR={sd}',
            f'mri_annotation2label --sd {sd} --subject {case} --hemi lh --lobesStrict lobes',
            f'mri_annotation2label --sd {sd} --subject {case} --hemi rh --lobesStrict lobes',
            f'mri_aparc2aseg --s {case} --labelwm --hypo-as-wm --rip-unknown --volmask --o "{wmparc_lobes}" --ctxseg aparc+aseg.mgz --annot lobes --base-offset 200',
        ]
        os.system('\n'.join(commands))
        assert wmparc_lobes.exists(), f'wmparc.lobes.mgz not found: "{wmparc_lobes}"'
    
    convert_mgz_to_nii(aparc_lobes)
    convert_mgz_to_nii(wmparc_lobes)
    convert_mgz_to_nii(brain)

    # Apply transform from MNI152 to MNI305
    jhu_stem = Path(jhu_mni152.stem).stem
    jhu_mni305 = (jhu_mni152.parent / f'{jhu_stem}_mni305.nii.gz')
    if not jhu_mni305.exists():
        mni_xfm = Path(args.mni_xfm)
        assert mni_xfm.exists(), f'MNI affine transformation not found: "{mni_xfm}"'
        mni305 = jhu_mni152.parent / 'mni305.cor.nii.gz'
        if not mni305.exists():
            mni305_mgz = '$FREESURFER_HOME/average/mni305.cor.mgz'
            os.system(f'mri_convert {mni305_mgz} "{mni305}"')
        assert mni305.exists(), f'MNI305 template not found: "{mni305}"'
        os.system(f'antsApplyTransforms -d 3 -i "{jhu_mni152}" -r "{mni305}" -o "{jhu_mni305}" -n NearestNeighbor -t "{mni_xfm}" -v 1')
        assert jhu_mni305.exists(), f'ANTs rigid-body transformation failed: "{jhu_mni305}"'
    
    # Transform IC and EC to conformed space
    jhu_conformed = mri_dir / f'{jhu_stem}_conformed.nii.gz'
    if not jhu_conformed.exists():
        xfm = mri_dir / 'transforms/talairach.xfm'
        orig = mri_dir / 'orig.mgz'
        os.system(f'mri_vol2vol --mov "{orig}" --targ "{jhu_mni305}" --o "{jhu_conformed}" --xfm "{xfm}" --inv --nearest')
        assert jhu_conformed.exists(), f'Talairach transformation failed: "{jhu_conformed}"'
    
    aparc_lobes = load_array(mri_dir / 'aparc.lobes.nii.gz')
    wmparc_lobes = load_array(mri_dir / 'wmparc.lobes.nii.gz')
    jhu_labels =  load_array(jhu_conformed)

    alloc_array = lambda: np.zeros_like(aparc_lobes, dtype=np.uint8)
    out_array = alloc_array()

    # Merge lobar labels
    lobar_labels = {
        1301: 'Frontal-Lobe',
        1303: 'Cingulate-Lobe',
        1304: 'Occipital-Lobe',
        1305: 'Temporal-Lobe',
        1306: 'Parietal-Lobe',
        1307: 'Insula-Lobe',
    }
    for id in list(lobar_labels.keys()):
        lobar_labels[id + 1000] = lobar_labels[id]
    for src_id in lobar_labels:
        out_array[aparc_lobes == src_id] = LOBAR_INDEX
    
    ctx_lobar_labels = dict(lobar_labels)
    lobar_labels = { src_id + 1900: name for src_id, name in ctx_lobar_labels.items() }
    for src_id in lobar_labels:
        out_array[wmparc_lobes == src_id] = LOBAR_INDEX
    
    lobar_labels = {
        17: 'Left-Hippocampus',
        18: 'Left-Amygdalae',
        53: 'Right-Hippocampus',
        54: 'Right-Amygdala',
    }
    for src_id in lobar_labels:
        out_array[wmparc_lobes == src_id] = LOBAR_INDEX

    # Merge deep supratentorial labels
    deep_labels = {
        10: 'Left-Thalamus',
        11: 'Left-Caudate',
        12: 'Left-Putamen',
        13: 'Left-Pallidum',
        49: 'Right-Thalamus',
        50: 'Right-Caudate',
        51: 'Right-Putamen',
        52: 'Right-Pallidum',
        251: 'CC_Posterior',
        252: 'CC_Mid_Posterior',
        253: 'CC_Central',
        254: 'CC_Mid_Anterior',
        255: 'CC_Anterior',
        26: 'Left-Accumbens-Area',
        58: 'Right-Accumbens-Area',
        28: 'Left-VentralDC',
        60: 'Right-VentralDC',
    }
    for src_id in deep_labels:
        out_array[aparc_lobes == src_id] = DEEP_INDEX
    deep_labels = {
        5001: 'Left-Deep-And-Periventricular-White-Matter',
        5002: 'Right-Deep-And-Periventricular-White-Matter',
    }
    dpwm_labels = alloc_array()
    for src_id in deep_labels:
        out_array[wmparc_lobes == src_id] = DEEP_INDEX
        dpwm_labels[wmparc_lobes == src_id] = 1
    jhu_wm_labels = {
        17: 'Internal-Capsule',
        18: 'Internal-Capsule',
        19: 'Internal-Capsule',
        20: 'Internal-Capsule',
    }
    ic_labels = alloc_array()
    for src_id in jhu_wm_labels:
        ic_labels[jhu_labels == src_id] = 1
    ic_labels = np.logical_and(dpwm_labels, ic_labels).astype(np.uint8)
    out_array[ic_labels == 1] = DEEP_INDEX
    jhu_wm_labels = {
        33: 'External-Capsule',
        34: 'External-Capsule',
    }
    ec_labels = alloc_array()
    for src_id in jhu_wm_labels:
        ec_labels[jhu_labels == src_id] = 1
    wm_insula = alloc_array()
    wm_labels = {
        3207: 'Left-Insula',
        4207: 'Right-Insula',
    }
    for src_id in wm_labels:
        wm_insula[wmparc_lobes == src_id] = 1
    ec_labels = np.logical_and(np.logical_or(dpwm_labels, wm_insula).astype(np.uint8), ec_labels).astype(np.uint8)
    out_array[ec_labels == 1] = DEEP_INDEX
    
    # Merge infratentorial labels
    infra_labels = {
        16: 'Brainstem',
        7: 'Cerebellum',
        8: 'Cerebellum',
        46: 'Cerebellum',
        47: 'Cerebellum',
    }
    for src_id in infra_labels:
        out_array[aparc_lobes == src_id] = INFRA_INDEX

    # Save labels
    src_image = sitk.ReadImage(str(mri_dir / 'brain.nii.gz'))
    out_image = sitk.GetImageFromArray(out_array)
    out_image.CopyInformation(src_image)
    if not args.output:
        out = mri_dir / 'proxy_label.nii.gz'
    else:
        out = Path(args.output)
    sitk.WriteImage(out_image, str(out))
    assert out.exists(), f'Failed to write target label: "{out}"'


def main():
    parser = argparse.ArgumentParser(description='Proxy Label Generation')
    parser.add_argument('--sd', type=str)
    parser.add_argument('--subject', type=str)
    parser.add_argument('--jhu_dti_atlas', type=str)
    parser.add_argument('--mni_xfm', type=str, default='mni152_to_mni305_GenericAffine.mat')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()
    generate_target_label(args)


if __name__ == '__main__':
    main()
