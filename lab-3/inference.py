import numpy
import supervision as sv
from PIL import Image
from streamlit import image, slider, toggle
from ultralytics import YOLO

validation_image_index = slider("Validation image index", 0, 99)
validation_image = Image.open(
    f"./data/CrowdPose/images/val/1000{validation_image_index:02}.jpg"
)

edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=1,
    edges=[
        (13, 14),
        (14, 1),
        (14, 2),
        (1, 3),
        (3, 5),
        (2, 4),
        (4, 6),
        (14, 8),
        (14, 7),
        (8, 10),
        (10, 12),
        (7, 9),
        (9, 11),
    ],
)
vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=2)


def show_annotated_image(key_points):
    annotated_image = validation_image.copy()
    edge_annotator.annotate(scene=annotated_image, key_points=key_points)
    vertex_annotator.annotate(scene=annotated_image, key_points=key_points)
    image(annotated_image)


with open(f"./data/CrowdPose/labels/val/1000{validation_image_index:02}.txt") as file:
    ground_truth = numpy.array(
        [[float(x) for x in line.split()[5:]] for line in file]
    ).reshape(-1, 14, 3)[:, :, 0:2]
    ground_truth[:, :, 0] *= validation_image.size[0]
    ground_truth[:, :, 1] *= validation_image.size[1]

ground_truth = sv.KeyPoints(
    ground_truth, class_id=numpy.array([0] * ground_truth.shape[0])
)

"Ground Truth"
show_annotated_image(ground_truth)


epoch = slider("Epoch", 0, 4)
confidence_threshold = slider("Confidence threshold", 0.0, 1.0, step=0.025)

with_augmentations = toggle("With augmentations")

results = YOLO(
    f"./runs/pose/train{10 if with_augmentations else 9}/weights/epoch{epoch}.pt"
)(validation_image)

assert len(results) == 1

predicted = sv.KeyPoints.from_ultralytics(results[0])
predicted.xy[predicted.confidence < confidence_threshold] = 0

"Prediction"
show_annotated_image(predicted)
