from datasets import load_dataset
import numpy as np
from PIL import Image

def load_data(image_size=(128, 128), max_per_class=1000):
    dataset = load_dataset("microsoft/cats_vs_dogs", split="train[:100%]")
    images = []
    labels = []

    cat_count = 0
    dog_count = 0

    for sample in dataset:
        label = sample['labels']
        if label == 0 and cat_count >= max_per_class:
            continue
        if label == 1 and dog_count >= max_per_class:
            continue

        pil_img = sample['image'].convert("L")
        img = pil_img.resize(image_size)
        img_np = np.array(img)
        images.append(img_np)
        labels.append(label)

        if label == 0:
            cat_count += 1
        else:
            dog_count += 1

        if cat_count >= max_per_class and dog_count >= max_per_class:
            break

    return np.array(images), np.array(labels)
