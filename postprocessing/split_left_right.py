# ============================================================
# SPLIT BILATERAL MUSCLE LABELS INTO ONE L/R MULTILABEL NIFTI
# ============================================================

from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------
# INPUT nnU-Net labels
# ------------------------------------------------------------
# 0 = background
# 1 = psoas
# 2 = quadratus lumborum
# 3 = erector spinae
# 4 = multifidus
# 5 = epifascial fat


# ------------------------------------------------------------
# OUTPUT labels
# ------------------------------------------------------------

OUTPUT_LABELS = {
    1: ("LPS", 1, 2),
    2: ("LQL", 3, 4),
    3: ("LES", 5, 6),
    4: ("LMF", 7, 8),
}

EPIFASCIAL_FAT_INPUT = 5
EPIFASCIAL_FAT_OUTPUT = 9


def world_x(affine, voxel):
    return nib.affines.apply_affine(
        affine,
        np.asarray(voxel, dtype=float)
    )[0]


def split_left_right(seg_img, label_value):

    data = np.asarray(seg_img.dataobj)
    mask = data == label_value

    left = np.zeros(mask.shape, dtype=bool)
    right = np.zeros(mask.shape, dtype=bool)

    axcodes = nib.aff2axcodes(seg_img.affine)

    try:
        si_axis = next(
            i for i, code in enumerate(axcodes)
            if code in ("S", "I")
        )
    except StopIteration:
        raise ValueError(
            f"Could not identify superior-inferior axis: {axcodes}"
        )

    mask_m = np.moveaxis(mask, si_axis, -1)
    left_m = np.moveaxis(left, si_axis, -1)
    right_m = np.moveaxis(right, si_axis, -1)

    remaining_axes = [
        i for i in range(3)
        if i != si_axis
    ]

    centre_voxel = (np.asarray(mask.shape) - 1) / 2

    image_mid_x = world_x(
        seg_img.affine,
        centre_voxel
    )

    for z in range(mask_m.shape[-1]):

        current = mask_m[..., z]

        if not np.any(current):
            continue

        components, n_components = ndimage.label(
            current,
            structure=np.ones((3, 3), dtype=np.uint8)
        )

        component_info = []

        for component_id in range(1, n_components + 1):

            coords = np.argwhere(
                components == component_id
            )

            if len(coords) == 0:
                continue

            centroid_2d = coords.mean(axis=0)

            voxel = np.zeros(3, dtype=float)

            voxel[remaining_axes[0]] = centroid_2d[0]
            voxel[remaining_axes[1]] = centroid_2d[1]
            voxel[si_axis] = z

            component_info.append({
                "id": component_id,
                "size": len(coords),
                "x": world_x(seg_img.affine, voxel),
            })

        if not component_info:
            continue

        largest = max(
            item["size"]
            for item in component_info
        )

        major = [
            item for item in component_info
            if item["size"] >= max(10, 0.10 * largest)
        ]

        if len(major) >= 2:

            xs = [item["x"] for item in major]

            split_x = (
                min(xs) + max(xs)
            ) / 2

        else:
            split_x = image_mid_x

        for item in component_info:

            component_mask = (
                components == item["id"]
            )

            # RAS coordinates:
            # negative/smaller x = anatomical LEFT
            # positive/larger x = anatomical RIGHT

            if item["x"] < split_x:
                left_m[..., z][component_mask] = True
            else:
                right_m[..., z][component_mask] = True

    return left, right


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert bilateral paraspinal muscle labels "
            "into one left/right multilabel NIfTI."
        )
    )

    parser.add_argument(
        "--seg",
        required=True,
        help="Input nnU-Net segmentation (.nii or .nii.gz)"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output multilabel NIfTI (.nii.gz)"
    )

    args = parser.parse_args()

    seg_path = Path(args.seg)
    output_path = Path(args.out)

    if not seg_path.exists():
        raise FileNotFoundError(seg_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    seg_img = nib.load(seg_path)

    data = np.asarray(seg_img.dataobj)

    output = np.zeros(
        data.shape,
        dtype=np.uint8
    )

    print(
        "NIfTI orientation:",
        nib.aff2axcodes(seg_img.affine)
    )

    # --------------------------------------------------------
    # Split four muscle groups
    # --------------------------------------------------------

    for input_label, (
        name,
        left_label,
        right_label
    ) in OUTPUT_LABELS.items():

        left, right = split_left_right(
            seg_img,
            input_label
        )

        output[left] = left_label
        output[right] = right_label

        print(
            f"{name}: "
            f"L={left_label}, R={right_label}"
        )

    # --------------------------------------------------------
    # Epifascial fat remains bilateral
    # --------------------------------------------------------

    output[
        data == EPIFASCIAL_FAT_INPUT
    ] = EPIFASCIAL_FAT_OUTPUT

    # --------------------------------------------------------
    # Save one multilabel segmentation
    # --------------------------------------------------------

    out_img = nib.Nifti1Image(
        output,
        seg_img.affine,
        seg_img.header
    )

    out_img.set_data_dtype(np.uint8)

    nib.save(
        out_img,
        output_path
    )

    print("\nSaved:")
    print(output_path)

    print("\nOutput labels:")
    print("0 = Background")
    print("1 = Left psoas")
    print("2 = Right psoas")
    print("3 = Left quadratus lumborum")
    print("4 = Right quadratus lumborum")
    print("5 = Left erector spinae")
    print("6 = Right erector spinae")
    print("7 = Left multifidus")
    print("8 = Right multifidus")
    print("9 = Epifascial fat")


if __name__ == "__main__":
    main()
