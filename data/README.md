# Dataset

The full labeled dataset used in this project is not included in this repository (research data, available on request for academic purposes).

## Expected structure

To use `src/train_yolo.py`, organize your dataset as follows and point `--data` to your `data.yaml`:

```
your_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**data.yaml:**
```yaml
train: your_dataset/images/train
val:   your_dataset/images/val
test:  your_dataset/images/test

nc: 3
names:
  - cubic
  - pseudo-cubic
  - ball
```

## Dataset details

| Split | Images |
|-------|--------|
| Train | ~900   |
| Val   | ~59    |
| Test  | ~32    |

- **Source:** SEM (Scanning Electron Microscope) images of calcium carbonate nanocrystals, acquired at Efrei INNOLab
- **3 classes:** `cubic`, `pseudo-cubic`, `ball`
- **Annotation pipeline:**
  1. Manual bounding-box annotation using `src/annotate.py annotate` → JSON
  2. JSON → YOLO format via `src/annotate.py convert`
  3. Synthetic augmentation with `src/data_augmentation.py` (patch pasting: random rotation, scale, position on SEM backgrounds)
- **Base images:** ~500 raw SEM .bmp images at two zoom levels
- **After augmentation:** ~1200 training images

## Requesting the dataset

Contact your institution/team for access to the full labeled dataset.
