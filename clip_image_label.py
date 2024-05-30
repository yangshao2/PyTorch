import rasterio
import numpy as np
import os

def clip_image_and_label(image_path, label_path, output_dir, patch_size=(256, 256)):
    # Create the output directories if they don't exist
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    with rasterio.open(image_path) as src_image:
        image = src_image.read()
        image_meta = src_image.meta.copy()
        image_transform = src_image.transform
        
    with rasterio.open(label_path) as src_label:
        label = src_label.read(1)  # Read the label as a single channel
        label_meta = src_label.meta.copy()
        label_transform = src_label.transform
    
    image_height, image_width = image.shape[1], image.shape[2]
    patch_height, patch_width = patch_size
    
    image_meta.update({
        'height': patch_height,
        'width': patch_width,
        'count': image.shape[0]  # Number of channels
    })
    
    label_meta.update({
        'height': patch_height,
        'width': patch_width,
        'count': 1  # Single channel
    })
    
    patch_id = 0
    for i in range(0, image_height, patch_height):
        for j in range(0, image_width, patch_width):
            image_patch = image[:, i:i + patch_height, j:j + patch_width]
            label_patch = label[i:i + patch_height, j:j + patch_width]
            
            # Ensure the patch is the correct size
            if image_patch.shape[1] == patch_height and image_patch.shape[2] == patch_width:
                # Calculate the transform for this patch
                new_transform = rasterio.transform.from_bounds(
                    image_transform * (j, i),
                    image_transform * (j + patch_width, i + patch_height),
                    patch_width,
                    patch_height
                )
                
                image_meta.update({'transform': new_transform})
                label_meta.update({'transform': new_transform})
                
                image_patch_filename = os.path.join(images_output_dir, f'image_patch_{patch_id}.tif')
                label_patch_filename = os.path.join(labels_output_dir, f'label_patch_{patch_id}.tif')
                
                with rasterio.open(image_patch_filename, 'w', **image_meta) as dst_image:
                    dst_image.write(image_patch)
                
                with rasterio.open(label_patch_filename, 'w', **label_meta) as dst_label:
                    dst_label.write(label_patch, 1)
                    
                print(f"Saved {image_patch_filename} and {label_patch_filename}")
                patch_id += 1

# Paths to the large image and label
image_path = '/home/yshao/unet/lc/stack.tif'
label_path = '/home/yshao/unet/lc/change.tif'
output_dir = '/home/yshao/unet/lc/newtrain'

clip_image_and_label(image_path, label_path, output_dir)

