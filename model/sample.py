import numpy as np
from torch import no_grad
import matplotlib.pyplot as plt

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def save_samples(adversary, noise, image_size, num_images=10, num_images_per_row=10, save_path=None):
    with no_grad():
        fake_images = adversary(noise).detach().cpu()

    fake_images = fake_images.numpy()
    fake_images = np.transpose(fake_images, (0, 2, 3, 1))
    fake_images = [normalize(img) for img in fake_images]

    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row
    num_cols = min(num_images, num_images_per_row)

    # Convert pixel dimensions to inches for figure size
    pixels_to_inches = 1 / 27  # This value might need adjustment for your display
    fig_width = num_cols * image_size * pixels_to_inches
    fig_height = num_rows * image_size * pixels_to_inches

    _, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for idx, img in enumerate(fake_images):
        if idx == num_images:
            break
        ax = axes[idx]
        ax.imshow(img)  # Removed aspect='auto'
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()  