import os

import pandas as pd
from tqdm import tqdm

from groundingdino.util.inference import load_image, load_model, predict

CONFIG_FILE = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"
DATASET_DIR_SINGLE_OBJECTS = "dataset/single_object"
DATASET_DIR_MULTIPLE_OBJECTS = "dataset/multi_object"

PROMPT_HIERARCHY_EXPERIMENT_1 = {
    "dog": ["dog", "animal", "mammal", "living thing"],
    "cat": ["cat", "animal", "mammal", "living thing"],
    "bird": ["bird", "animal", "living thing"],
    "car": ["car", "vehicle", "object"],
    "bicycle": ["bicycle", "vehicle", "object"],
    "chair": ["chair", "furniture", "object"],
}

MULTI_OBJECT_PROMPTS = ["object", "thing", "animal", "vehicle"]


def run_groundingdino(
    model, image_path, prompt, box_threshold=0.35, text_threshold=0.25
):
    _, image_tensor = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu",
    )
    return boxes, logits, phrases


if __name__ == "__main__":
    model = load_model(CONFIG_FILE, CHECKPOINT_PATH, device="cpu")

    # Experiment 1
    results_experiment_1 = []
    for cls, prompts in tqdm(PROMPT_HIERARCHY_EXPERIMENT_1.items(), desc="Classes"):
        class_dir = os.path.join(DATASET_DIR_SINGLE_OBJECTS, cls)
        image_dir = os.path.join(class_dir, "data")

        image_files = os.listdir(image_dir)
        for img_name in tqdm(image_files, desc=f"Images ({cls})", leave=False):
            img_path = os.path.join(class_dir, "data", img_name)

            for prompt in prompts:
                boxes, logits, phrases = run_groundingdino(model, img_path, prompt)

                results_experiment_1.append(
                    {
                        "class": cls,
                        "image": img_name,
                        "prompt": prompt,
                        "num_detections": len(boxes),
                        "max_confidence": (
                            float(logits.max()) if len(logits) > 0 else 0.0
                        ),
                    }
                )

    results_experiment_1_df = pd.DataFrame(results_experiment_1)
    results_experiment_1_df.to_csv("experiment_1_results.csv", index=False)

    results_experiment_2 = []
    image_dir = os.path.join(DATASET_DIR_MULTIPLE_OBJECTS, "data")
    image_files = os.listdir(image_dir)

    for img_name in tqdm(image_files, desc="Multi-object images"):
        img_path = os.path.join(DATASET_DIR_MULTIPLE_OBJECTS, "data", img_name)

        for prompt in MULTI_OBJECT_PROMPTS:
            boxes, logits, phrases = run_groundingdino(model, img_path, prompt)

            results_experiment_2.append(
                {
                    "image": img_name,
                    "prompt": prompt,
                    "num_detections": len(boxes),
                    "avg_confidence": float(logits.mean()) if len(logits) > 0 else 0.0,
                }
            )

    results_experiment_2_df = pd.DataFrame(results_experiment_2)
    results_experiment_2_df.to_csv("experiment_2_results.csv", index=False)
