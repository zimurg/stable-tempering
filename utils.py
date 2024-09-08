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


def save_latents (latents, prompt="latents", seed=0, step=None):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    
    if len(prompt) < 16:
        folder_name = prompt + ' ' * (16 - len(prompt))
    else:
        folder_name = prompt[:16]
    
    folder_name = folder_name.strip().replace(' ', '_') + f"_{seed}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    latent_path = os.path.join(folder_name, f"latents_{seed}_step{step}.pt")
    torch.save(latents, latent_path)


def save_image (image, prompt="image", seed=0, step=None):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    
    if len(prompt) < 16:
        folder_name = prompt + ' ' * (16 - len(prompt))
    else:
        folder_name = prompt[:16]
    
    folder_name = folder_name.strip().replace(' ', '_')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if step is not None:
        image_name = f"image_{folder_name}_step{step}_{seed}.png"
    else:
        image_name = f"image_{folder_name}_{seed}.png"
    
    image_path = os.path.join(folder_name, image_name)
    image.save(image_path)
