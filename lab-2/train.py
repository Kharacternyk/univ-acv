import json

from numpy import array
from PIL import Image
from torch import optim, save, stack
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessorFast,
)
from transformers.image_transforms import rgb_to_id

import wandb


class Cityscapes(Dataset):
    def __init__(self, subset: str):
        with open(f"./data/gtFine/cityscapes_panoptic_{subset}_trainId.json") as file:
            metadata = json.load(file)

        self.annotations = metadata["annotations"]
        self.processor = Mask2FormerImageProcessorFast(ignore_index=0)
        self.subset = subset

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        image_id = item["image_id"]
        city = image_id.split("_")[0]

        image = Image.open(
            f"./data/leftImg8bit/{self.subset}/{city}/{image_id}_leftImg8bit.png"
        )
        panoptic = Image.open(
            f"./data/gtFine/cityscapes_panoptic_{self.subset}_trainId"
            f"/{image_id}_gtFine_panoptic.png"
        )

        segments = item["segments_info"]
        category_mapping = {
            segment["id"]: segment["category_id"] for segment in segments
        }

        mask = rgb_to_id(array(panoptic))

        features = self.processor(image, mask, category_mapping)

        return [
            features[name][0]
            for name in ["pixel_values", "mask_labels", "class_labels"]
        ] + [image]

    def create_dataloader(self, **kwargs):
        def collate(samples):
            return [
                stack([sample[0] for sample in samples]),
                [sample[1] for sample in samples],
                [sample[2] for sample in samples],
            ]

        return DataLoader(self, collate_fn=collate, **kwargs)


categories = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-panoptic",
    id2label=categories,
    ignore_mismatched_sizes=True,
)
model.cuda()

if __name__ != "__main__":
    model.eval()
else:
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    dataloader = Cityscapes("train").create_dataloader(
        batch_size=4, num_workers=4, shuffle=True
    )

    with wandb.init(project="univ-acv-lab-2") as logger:
        for epoch in range(5):
            for index, batch in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()

                pixels, mask_labels, class_labels, *_ = batch
                pixels = pixels.to(model.device)
                mask_labels = [labels.to(model.device) for labels in mask_labels]
                class_labels = [labels.to(model.device) for labels in class_labels]

                output = model(pixels, mask_labels, class_labels)

                output.loss.backward()
                optimizer.step()

                global_index = epoch * len(dataloader) + index + 1

                if not global_index % 5:
                    logger.log(dict(loss=output.loss), step=global_index)

                if not global_index % 100:
                    save(model.state_dict(), f"checkpoint_{global_index:05}.pt")
