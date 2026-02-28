from collections import defaultdict

import streamlit
import torch
from matplotlib import cm, pyplot
from matplotlib.patches import Patch
from streamlit import cache_resource, expander, image, slider
from torch import from_numpy, inference_mode, load, unique, zeros

from train import Cityscapes, categories, model


@cache_resource
def dataset():
    return Cityscapes("val")


train_batch_count = slider("Number of training batches", 100, 2500, step=100)
validation_sample_index = slider("Index of validation sample", 0, len(dataset()) - 1)

model.load_state_dict(load(f"checkpoint_{train_batch_count:05}.pt"))

sample = dataset()[validation_sample_index]

with expander("Original image"):
    image(sample[-1])


def show_overlay(overlay, labels=categories):
    figure, axes = pyplot.subplots()
    cmap = "gist_ncar"

    axes.imshow(overlay, vmax=len(labels) - 1, cmap=cmap)
    axes.set_axis_off()

    colors = cm.get_cmap(cmap, len(labels))

    patches = [
        Patch(color=colors(int(i)), label=labels[int(i)])
        for i in unique(overlay)
        if labels[int(i)]
    ]
    axes.legend(handles=patches, bbox_to_anchor=(0, 1))

    streamlit.pyplot(figure)


target_sizes = [(sample[-1].size[1], sample[-1].size[0])]

with inference_mode():
    output = model(sample[0].unsqueeze(0).to(model.device))

    with expander("Semantic segmentation"):
        result = dataset().processor.post_process_semantic_segmentation(
            output, target_sizes=target_sizes
        )[0]
        show_overlay(result.cpu())

    with expander("Instance segmentation"):
        result = dataset().processor.post_process_instance_segmentation(
            output, target_sizes=target_sizes
        )[0]

        labels = ["background"]

        for segment in result["segments_info"]:
            labels.append(categories[segment["label_id"]])

        show_overlay(result["segmentation"].cpu() + 1, labels)

    with expander("Panoptic segmentation"):
        result = dataset().processor.post_process_panoptic_segmentation(
            output, target_sizes=target_sizes
        )[0]

        labels = [None]

        for segment in result["segments_info"]:
            labels.append(categories[segment["label_id"]])

        show_overlay(result["segmentation"].cpu(), labels)

overlay = zeros(sample[1].shape[1:])
labels = ["other"]

for i in range(1, sample[1].shape[0]):
    mask = sample[1][i] == 1.0

    if mask.any():
        overlay[mask] = i
        labels.append(categories[sample[2][i]])

with expander("Panoptic ground truth"):
    show_overlay(overlay, labels)
