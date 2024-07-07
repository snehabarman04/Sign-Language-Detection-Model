import os
import json
from pathlib import Path

base_dir = Path(__file__).parent
config_path = base_dir / 'config.json'

with open(config_path, 'r') as config_file:
    config = json.load(config_file)

train_dir = base_dir / config['train_dir']
test_dir = base_dir / config['test_dir']

def count_images_in_directory(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    return class_counts

train_counts = count_images_in_directory(train_dir)
test_counts = count_images_in_directory(test_dir)

print("Training image counts per class:", train_counts)
print("Testing image counts per class:", test_counts)
