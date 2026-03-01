from os import getenv

import albumentations as A
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
model.train(
    data="./crowdpose.yaml",
    epochs=5,
    batch=32,
    cache=True,
    workers=4,
    deterministic=False,
    save_period=1,
    # Disable built-in augmentations
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    degrees=0,
    translate=0,
    scale=0,
    fliplr=0,
    mosaic=0,
    # Enable custom augmentations
    augmentations=[]
    if not getenv("ENABLE_AUG")
    else [
        A.RandomOrder(
            [
                A.AdvancedBlur(),
                A.RandomBrightnessContrast(),
                A.RandomToneCurve(),
                A.Rotate(),
            ],
            n=2,
            p=0.5,
        ),
    ],
)
