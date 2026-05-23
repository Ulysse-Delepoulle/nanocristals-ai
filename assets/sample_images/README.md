# Sample Images

This folder contains example SEM images illustrating the three nanocrystal classes detected by the model.

## Classes

| Class | Description |
|-------|-------------|
| `cubic` | Well-defined cubic crystal morphology |
| `pseudo-cubic` | Intermediate morphology between cubic and spherical |
| `ball` | Spherical / amorphous nanoparticle |

## Adding your own samples

Place representative images here, ideally 2-3 per class. Accepted formats: `.jpg`, `.png`.

To run inference on any image:
```bash
python src/predict.py --image assets/sample_images/your_image.jpg
```
