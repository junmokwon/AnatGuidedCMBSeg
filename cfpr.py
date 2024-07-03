from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk


LOBAR_INDEX = 1
DEEP_INDEX = 2
INFRA_INDEX = 3
CMB_INDEX = 4


def binarize(src_image, lower, upper, val=1):
    bf = sitk.BinaryThresholdImageFilter()
    bf.SetLowerThreshold(lower)
    bf.SetUpperThreshold(upper)
    bf.SetInsideValue(val)
    bf.SetOutsideValue(0)
    return bf.Execute(src_image)


def connected_component(image):
    filter = sitk.ConnectedComponentImageFilter()
    filter.FullyConnectedOn()
    output_image = filter.Execute(image)
    num_components = filter.GetObjectCount()
    return output_image, num_components


def cfpr(args):
    pred_file = Path(args.input)
    assert pred_file.exists(), f'Prediction file not found: "{pred_file}"'
    pred_image = sitk.ReadImage(str(pred_file))
    pred_array = sitk.GetArrayFromImage(pred_image)
    proxy_indices = [LOBAR_INDEX, DEEP_INDEX, INFRA_INDEX]
    proxy_image = binarize(pred_image, min(proxy_indices), max(proxy_indices))
    proxy_array = sitk.GetArrayFromImage(proxy_image)
    cmb_image = binarize(pred_image, CMB_INDEX, CMB_INDEX)
    cmb_conn_image, n_cmb_regions = connected_component(cmb_image)
    proxy_array = sitk.GetArrayFromImage(proxy_image)
    for k in range(1, n_cmb_regions + 1):
        mask_image = binarize(cmb_conn_image, k, k)
        cmb_dist_image = sitk.SignedMaurerDistanceMap(mask_image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        cmb_dist_lower = binarize(cmb_dist_image, 0, args.dilate_mm)
        xor_filter = sitk.XorImageFilter()
        peri_image = xor_filter.Execute(mask_image, cmb_dist_lower)
        peri_array = sitk.GetArrayFromImage(peri_image)
        peri_count = np.count_nonzero(peri_array)
        brain_parenchyma_ratio = np.count_nonzero(proxy_array[peri_array == 1]) / peri_count
        if brain_parenchyma_ratio < args.brain_parenchyma_threshold:
            mask_array = sitk.GetArrayFromImage(mask_image)
            pred_array[mask_array == 1] = 0
    out_image = sitk.GetImageFromArray(pred_array)
    out_image.CopyInformation(pred_image)
    out_file = Path(args.output)
    sitk.WriteImage(out_image, str(out_file))
    assert out_file.exists(), f'Failed to write CFPR: "{out_file}"'


def main():
    parser = argparse.ArgumentParser(description='Clinically-derived False Positive Reduction')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--brain_parenchyma_threshold', type=float, default=0.5)
    parser.add_argument('--dilate_mm', type=float, default=2.0)
    args = parser.parse_args()
    cfpr(args)


if __name__ == '__main__':
    main()
