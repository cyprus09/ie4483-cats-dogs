import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

class Augmentation:
    def __init__(self, base_transform, mixup_alpha=1.0, p_mixup=0.5, p_mosaic=0.3):
        self.base_transform = base_transform
        self.mixup_alpha = mixup_alpha
        self.p_mixup = p_mixup
        self.p_mosaic = p_mosaic
        
    def __call__(self, img):
        """Make the class callable for use as a transform"""
        # Apply base transforms first
        img_tensor = self.base_transform(img)
        return img_tensor
        
    def mixup(self, batch, labels):
        """Applies mixup to a batch of images"""
        if np.random.random() > self.p_mixup:
            return batch, labels, labels, 1.0
            
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = batch.size()[0]
        index = torch.randperm(batch_size)

        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        return mixed_batch, labels, labels[index], lam

    def mosaic(self, img, dataset):
        """Applies mosaic augmentation to a single image"""
        if np.random.random() > self.p_mosaic:
            return self.base_transform(img)
            
        # Get 3 more random images
        indices = np.random.randint(0, len(dataset), 3)
        imgs = [dataset[i][0] for i in indices]  # Get just the images
        imgs = [img] + imgs  # Add original image
        
        size = 227  # Same as your current size
        result = Image.new('RGB', (size * 2, size * 2))
        
        # Random center point
        cx = int(random.uniform(size * 0.5, size * 1.5))
        cy = int(random.uniform(size * 0.5, size * 1.5))
        
        for i, img in enumerate(imgs):
            if isinstance(img, torch.Tensor):
                # Convert tensor back to PIL Image if needed
                img = transforms.ToPILImage()(img)
                
            if i == 0:  # Top-left
                x1, y1 = 0, 0
                x2, y2 = cx, cy
            elif i == 1:  # Top-right
                x1, y1 = cx, 0
                x2, y2 = size * 2, cy
            elif i == 2:  # Bottom-left
                x1, y1 = 0, cy
                x2, y2 = cx, size * 2
            else:  # Bottom-right
                x1, y1 = cx, cy
                x2, y2 = size * 2, size * 2
            
            w = x2 - x1
            h = y2 - y1
            resized_img = img.resize((w, h), Image.Resampling.BILINEAR)
            result.paste(resized_img, (x1, y1))
        
        result = result.resize((size, size), Image.Resampling.BILINEAR)
        return self.base_transform(result)