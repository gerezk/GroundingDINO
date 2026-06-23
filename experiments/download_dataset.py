import os
import random

import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np

EXPORT_DIR = "dataset"
LABEL_FIELD = "ground_truth"
DATASET_TYPE = fo.types.COCODetectionDataset
CLASSES = ["dog", "cat", "bird", "car", "bicycle", "chair", "apple"]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    classes=CLASSES,
    max_samples=1000,
    shuffle=True,
    overwrite=True,
    seed=SEED,
)

# ----------
# Create Single object data
# ----------
single_object_view = dataset.match(
    fo.ViewField("ground_truth.detections").length() <= 2
)
for cls in CLASSES:
    view = (
        single_object_view.filter_labels(LABEL_FIELD, fo.ViewField("label") == cls)
        .shuffle(seed=SEED)
        .take(10)
    )

    view.export(
        export_dir=os.path.join(EXPORT_DIR, "single_object", cls),
        dataset_type=DATASET_TYPE,
        label_field=LABEL_FIELD,
    )

# ----------
# Create Multi object data
# ----------
multi_object_view = (
    dataset.match(fo.ViewField("ground_truth.detections").length() >= 3)
    .shuffle(seed=SEED)
    .take(30)
)
multi_object_view.export(
    export_dir=os.path.join(EXPORT_DIR, "multi_object"),
    dataset_type=DATASET_TYPE,
    label_field=LABEL_FIELD,
)
