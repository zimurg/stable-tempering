import torch
import numpy as np
from PIL import Image
import os

def image_from_tensor(tensor):
    tensor = tensor.permute(0, 2, 3, 1).contiguous()
    tensor = (tensor * 255).clamp(0, 255).byte()
    array = tensor.cpu().numpy()[0].astype(np.uint8)
    image = Image.fromarray(array)
    return image


def save_latents (latents, step, prompt="latents", seed=0):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    
    if len(prompt) < 16:
        folder_name = prompt + ' ' * (16 - len(prompt))
    else:
        folder_name = prompt[:16]
    
    folder_name = folder_name.strip().replace(' ', '_') + f"_{seed}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    image_path = os.path.join(folder_name, f"image_{step}.png")
    latents.save(image_path)


def save_image (image, prompt="image", seed=0):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    
    if len(prompt) < 16:
        folder_name = prompt + ' ' * (16 - len(prompt))
    else:
        folder_name = prompt[:16]
    
    folder_name = folder_name.strip().replace(' ', '_')
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    image_path = os.path.join(folder_name, f"image_{folder_name}.png")
    image.save(image_path)
