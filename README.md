# Open-Source Paraspinal Muscle Segmentation on Lumbar MRI Using nnU-Net
Open-source nnU-Net model for automated segmentation of lumbar paraspinal muscles and epifascial fat on axial T2-weighted MRI.

## Segmented structures

The model provides separate segmentations of:

- Psoas
- Quadratus lumborum
- Erector spinae
- Multifidus
- Epifascial fat

### Label map

| Label | Structure |
|------:|-----------|
| 0 | Background |
| 1 | Psoas |
| 2 | Quadratus lumborum |
| 3 | Erector spinae |
| 4 | Multifidus |
| 5 | Epifascial fat |

## Model

The final model is based on **nnU-Net v2** using a 2D configuration.

It was developed and evaluated using participant-level five-fold cross-validation on axial T2-weighted lumbar MRI.

## Availability

Trained model weights and instructions for inference will be provided in this repository.

## Citation

Publication information will be added following publication.

## Intended use

The model is provided for research use. Independent validation is recommended before application to substantially different imaging protocols or clinical populations.
