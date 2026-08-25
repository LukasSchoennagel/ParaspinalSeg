# ============================================================
# SPLIT BILATERAL MUSCLE LABELS INTO LEFT / RIGHT
# ============================================================

from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------
# nnU-Net label map
# ------------------------------------------------------------

LABELS = {
    1: "PS",   # Psoas
    2: "QL",   # Quadratus lumborum
    3: "ES",   # Erector spinae
    4: "MF",   # Multifidus
}

EPIFASCIAL_FAT_LABEL = 5


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def world_x(affine, voxel):
    """Return RAS world-coordinate x position."""
    return nib.affines.apply_affine(
        affine,
        np.asarray(voxel, dtype=float)
    )[0]


def split_left_right(seg_img, label_value):

    data = np.asarray(seg_img.dataobj)
    mask = data == label_value

    left = np.zeros(mask.shape, dtype=np.uint8)
    right = np.zeros(mask.shape, dtype=np.uint8)

    # Determine superior-inferior array axis from affine
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

    # Move slice axis to the final dimension
    mask_m = np.moveaxis(mask, si_axis, -1)
    left_m = np.moveaxis(left, si_axis, -1)
    right_m = np.moveaxis(right, si_axis, -1)

    remaining_axes = [
        i for i in range(3)
        if i != si_axis
    ]

    # World-coordinate position of image centre.
    centre_voxel = (np.asarray(mask.shape) - 1) / 2
    image_mid_x = world_x(
        seg_img.affine,
        centre_voxel
    )

    for z in range(mask_m.shape[-1]):

        current = mask_m[..., z]

        if not np.any(current):
            continue

        # 8-connected components within the axial slice
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

        # ----------------------------------------------------
        # Estimate left/right division
        # ----------------------------------------------------

        largest = max(
            item["size"]
            for item in component_info
        )

        # Ignore tiny islands when estimating the midline
        major = [
            item for item in component_info
            if item["size"] >= max(10, 0.10 * largest)
        ]

        if len(major) >= 2:

            xs = [
                item["x"]
                for item in major
            ]

            split_x = (
                min(xs) + max(xs)
            ) / 2

        else:
            # If only one component is visible, use image centre
            split_x = image_mid_x

        # In RAS coordinates:
        # smaller x = anatomical LEFT
        # larger x  = anatomical RIGHT

        for item in component_info:

            component_mask = (
                components == item["id"]
            )

            if item["x"] < split_x:
                left_m[..., z][component_mask] = 1
            else:
                right_m[..., z][component_mask] = 1

    return left, right


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Split bilateral paraspinal muscle labels "
            "into anatomical left and right masks."
        )
    )

    parser.add_argument(
        "--seg",
        required=True,
        help="nnU-Net segmentation (.nii or .nii.gz)"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    seg_path = Path(args.seg)
    output_dir = Path(args.out)

    if not seg_path.exists():
        raise FileNotFoundError(seg_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    seg_img = nib.load(seg_path)

    print(
        "NIfTI orientation:",
        nib.aff2axcodes(seg_img.affine)
    )

    # --------------------------------------------------------
    # Split muscles
    # --------------------------------------------------------

    for label_value, abbreviation in LABELS.items():

        left, right = split_left_right(
            seg_img,
            label_value
        )

        for side, mask in [
            ("L", left),
            ("R", right),
        ]:

            output_path = (
                output_dir /
                f"{side}{abbreviation}.nii.gz"
            )

            out_img = nib.Nifti1Image(
                mask,
                seg_img.affine,
                seg_img.header
            )

            out_img.set_data_dtype(np.uint8)

            nib.save(
                out_img,
                output_path
            )

            print(
                f"Saved: {output_path}"
            )

    # --------------------------------------------------------
    # Epifascial fat remains bilateral / total
    # --------------------------------------------------------

    data = np.asarray(seg_img.dataobj)

    fat = (
        data == EPIFASCIAL_FAT_LABEL
    ).astype(np.uint8)

    fat_img = nib.Nifti1Image(
        fat,
        seg_img.affine,
        seg_img.header
    )

    fat_img.set_data_dtype(np.uint8)

    fat_path = (
        output_dir /
        "FA.nii.gz"
    )

    nib.save(
        fat_img,
        fat_path
    )

    print(
        f"Saved: {fat_path}"
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()
