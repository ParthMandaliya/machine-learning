import cv2
import numpy as np
import os

def calculate_mean_std(image_dir):
    means = []
    stds = []
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None: #check if the image was read correctly
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to RGB
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        else:
            print(f"Error reading image: {image_path}")
    
    means = np.array(means).mean(axis=0)
    stds = np.array(stds).mean(axis=0)
    return np.round(means/255.0, 3).tolist(), np.round(stds/255.0, 3).tolist()

# Example usage (replace with your image directory)
image_directory = "dataset/test"
mean, std = calculate_mean_std(image_directory)
print(image_directory)
print(f"{mean=}")
print(f"{std=}")
