import os

from torch import tensor, int32
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize, Lambda
from torch.utils.data import Dataset, DataLoader

from cv2 import imread, cvtColor, IMREAD_UNCHANGED, COLOR_BGRA2RGBA, COLOR_BGR2RGB

from PIL.Image import ANTIALIAS, BICUBIC
from PIL.ImageOps import crop, pad, expand

from PIL import Image as PILImage

import matplotlib.pyplot as plt


def fit_image(image, target_height, target_width, method="resize"):
    assert method == "resize" or method == "crop"
    
    if method == "resize":
        image.thumbnail((target_width, target_height), ANTIALIAS)
    
    delta_width = target_width - image.size[0]
    delta_height = target_height - image.size[1]
    
    if method == "resize":
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        image = expand(image, padding, fill=(0, 0, 0, 0))
    else:
        padding = (delta_width, delta_height, delta_width, delta_height)
        image = expand(image, padding, fill=(0, 0, 0, 0))
        image = crop(image, (target_height, target_width))
        
        if image.size[0] != target_width or image.size[1] != target_height:
            image = pad(image, (target_width, target_height), centering=(0.5, 0.5))
    
    return image
        

def get_transformer(image_size, num_channels):
    assert num_channels == 3 or num_channels == 4
    
    channel_norm = (0.5, 0.5, 0.5) if num_channels == 3 else (0.5, 0.5, 0.5, 0.5)
    
    return Compose([
        # Resize(min(height_width), interpolation=BICUBIC),
        # CenterCrop(height_width),
        Lambda(lambda image: fit_image(image, image_size, image_size, method="resize")),
        ToTensor(),
        Normalize(channel_norm, channel_norm),
    ])
    
    
class Data(Dataset):
    def __init__(self, data_directory, category, num_channels, transform=None, blacklist=[]):
        self.data_directory = data_directory
        self.transform = transform
        self.category = category
        self.num_channels = num_channels
        self.ids = []
    
        for file_name in os.listdir(data_directory):
            if file_name.endswith(".png"):
                _id = int(file_name.split(".")[0])
                if _id not in blacklist:
                    self.ids.append(_id)
                

    def __str__(self):
        return f"{self.category.name} ({len(self.ids)} images)"

    
    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        image_name = f"{self.ids[idx]}.png"
        image_name = "{:0>11}".format(image_name)
        image_path = os.path.join(self.data_directory, image_name)
        image = None
        
        if self.num_channels == 3:
            image = imread(image_path)
            image = cvtColor(image, COLOR_BGR2RGB)
        else:
            image = imread(image_path, IMREAD_UNCHANGED)
            image = cvtColor(image, COLOR_BGRA2RGBA)
        
        
        image = PILImage.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        image = image * 0.5 + 0.5
        image = image.clone().detach()
        label = tensor(self.category.value, dtype=int32)
        
        return image, label
    

def show_samples(self, num_samples, num_samples_per_row=10, num_rows=1):
    num_rows = (num_samples // num_samples_per_row)
    plt.figure(figsize=(num_samples_per_row * 3, num_rows * 3))
    
    
    total_samples = num_samples_per_row * num_rows
    sample_count = 0

    for batch, _ in self:
        for i in range(len(batch)):
            if sample_count == total_samples:
                break
            
            img = batch[i].squeeze()  # Squeeze the individual image
            img = img.cpu().numpy().transpose(1, 2, 0)  # Convert to NumPy and transpose

            plt.subplot(num_rows, num_samples_per_row, sample_count + 1)
            plt.imshow(img)
            plt.axis('off')

            sample_count += 1

        if sample_count == total_samples:
            break

    plt.show()

# Monkey patch the DataLoader class to add the show_samples method
DataLoader.show_samples = show_samples