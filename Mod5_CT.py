#!/usr/bin/env python
# coding: utf-8

import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms


# Define augmentation functions
def augment_image(image):
    """Apply a random augmentation to the given image."""
    augmentations = [
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5)),  # Adjust brightness
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5)),    # Adjust contrast
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),                           # Horizontal flip
        lambda img: img.rotate(random.choice([15, -15, 30, -30])),                  # Rotate
        lambda img: transforms.RandomResizedCrop(size=img.size, scale=(0.8, 1.0))(transforms.ToTensor()(img)),  # Crop and resize
    ]

    augmentation = random.choice(augmentations)
    return augmentation(image)

# Define function to process the dataset
def augment_dataset(input_dir, output_dir, augmentations_per_image=1):
    """Augment all images in the input directory and save them to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert("RGB")

            for i in range(augmentations_per_image):
                augmented_image = augment_image(image)
                augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                augmented_image_path = os.path.join(output_dir, augmented_filename)

                if isinstance(augmented_image, Image.Image):
                    augmented_image.save(augmented_image_path)
                else:
                    transforms.ToPILImage()(augmented_image).save(augmented_image_path)


# Run the script
if __name__ == "__main__":
    input_folder = "galaxy"  # Folder containing the original images
    output_folder = "augmented_galaxy"  # Folder to save augmented images

    augment_dataset(input_folder, output_folder)
    print(f"Augmentation complete. Check the '{output_folder}' folder for results.")





