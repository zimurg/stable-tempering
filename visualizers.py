import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_segmentation(image, logits, prompts):
    preds = torch.sigmoid(logits).unsqueeze(1)  # Alplicando sigmoide para convertir logits en probabilidades
    _, ax = plt.subplots(1, len(prompts) + 1, figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(image)
    for i in range(len(prompts)):
        ax[i + 1].imshow(preds[i][0].cpu().numpy())
        ax[i + 1].text(0, -15, prompts[i])
    plt.show()

def visualize_masks(masks_output):
    masks = masks_output['masks']
    class_labels = masks_output['class_labels']
    unique_classes = masks.unique()

    cols = len(unique_classes)+1
    fig, ax = plt.subplots(1, cols, figsize=(15, 5))
    [a.axis('off') for a in ax]

    # Añadimos una etiqueta para la región no clasificada (al no ser un algoritmo "greedy"), si existe.
    if -1 in unique_classes:
        ax[0].set_title("Sin clasificar")
        idx_shift = 1  # Valor de offset si existen píxels no clasificados
    else:
        idx_shift = 0

    for idx, val in enumerate(unique_classes):
        if val == -1:
            mask_image = (masks == val).int()  # Máscara binaria para píxels no clasificados
            ax[idx].imshow(mask_image, cmap='gray')
        else:
            mask_image = (masks == val).int()  # Máscara binaria para la clase actual
            ax[idx + idx_shift].imshow(mask_image, cmap='gray')
            ax[idx + idx_shift].set_title(class_labels[val])

    plt.show()


def image_from_tensor(tensor):
    tensor = tensor.permute(0, 2, 3, 1).contiguous()
    tensor = (tensor * 255).clamp(0, 255).byte()
    array = tensor.cpu().numpy()[0].astype(np.uint8)
    image = Image.fromarray(array)
    return image

############################################################################################################
#
# La siguiente función está basada en Vass(2024)
# Vass, Timothy Alexis(2024). Explaining the SDXL latent space. En https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
#
############################################################################################################

def visualize_latents(latents): #TODO: citar la fuente de este snippet
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )
    
    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)
    
    Image.fromarray(image_array).show()
    return Image.fromarray(image_array)