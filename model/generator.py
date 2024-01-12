from model.adversary import Adversary, AdversaryType
from torch.nn import Linear
from torch.cuda.amp import autocast
from torch import full, zeros, FloatTensor, save
from torch.nn.utils import clip_grad_norm_
from torch import no_grad
import numpy as np
from matplotlib import pyplot as plot

class Generator(Adversary):
    def __init__(self, noise_size, image_size, channel_size, base_width):
        super().__init__(AdversaryType.Generator, noise_size, image_size, channel_size, base_width)
        self.linear_layer = None
    
    def __repr__(self):
        return f"Generator = {super().__repr__()}"
    
    def show_samples(self, noise, num_images=5):
        def normalize(image):
            min_val = np.min(image)
            max_val = np.max(image)
            return (image - min_val) / (max_val - min_val)

        with no_grad():
            fake_images = self(noise).detach().cpu()

        fake_images = fake_images.numpy()
        fake_images = np.transpose(fake_images, (0, 2, 3, 1))
        fake_images = [normalize(img) for img in fake_images]

        num_rows = (num_images + 7) // 8
        num_cols = min(num_images, 8)

        # Convert pixel dimensions to inches for figure size
        pixels_to_inches = 1 / 27  # This value might need adjustment for your display
        fig_width = num_cols * self.image_size * pixels_to_inches
        fig_height = num_rows * self.image_size * pixels_to_inches

        _, axes = plot.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        for idx, img in enumerate(fake_images):
            if idx == num_images:
                break
            ax = axes[idx]
            ax.imshow(img)  # Removed aspect='auto'
            ax.axis('off')

        for ax in axes[num_images:]:
            ax.axis('off')

        plot.show()  
    
    
    def add_linear_layer(self, out_factor):  
        assert len(self.layers) == 0

        output_size = self.base_width * out_factor * self.final_size * self.final_size
        self.linear_layer = Linear(self.noise_size, output_size)
        
    
    def forward(self, input):
        if self.linear_layer is not None:
            input = self.linear_layer(input).view(-1, self.base_width * 8, self.final_size, self.final_size)
        
        return self.layers(input)