import matplotlib.pyplot as plt
import numpy as np
from CNN.load_and_process_data import train_generator

# Get a batch of images and labels
x_batch, y_batch = next(train_generator)

# Define class names for better readability
class_names = train_generator.class_indices
class_names = {v: k for k, v in class_names.items()}  # Reversing the dictionary

# Display 9 images from the batch
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])
    plt.title(f"Class: {class_names[np.argmax(y_batch[i])]}")
    plt.axis('off')
plt.show()
