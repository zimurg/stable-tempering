import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image


def segment_image(image, prompts, processor, model):
    """
    Segmenta la imagen con CLIPSeg.
    
    Args:
    image (Image.Image): Imagen en formato PIL.Image. Es probable que acepte otros tipos, como PNG.
    prompts (list of str): Cada entrada se corresponde con una de las clases a segmentar.
    processor (CLIPSegProcessor): Modelo entrenado para decodificar los prompts en embeddings de CLIP
    model (CLIPSegForImageSegmentation): Modelo entrenado que toma las salidas del processor, y segmenta la imagen en base a ellas.

    Salidas:
    logits (torch.Tensor): Los logits devueltos por el modelo. Shape [num_classes, 1, height, width].
    """    
    original_size = image.size
    processor.image_processor.size = original_size #Aseguramos que el tamaño de los logits coincida con la imagen

    inputs = processor(text=prompts, images=[image] * len(prompts), return_tensors="pt")
    print("Tamaño de la imagen procesada:", inputs["pixel_values"].shape)

    with torch.no_grad():
        outputs = model(**inputs)

    if outputs.logits.shape[-2:] != original_size: #En caso de que no coincidan los tamaños de máscaras e imagen a la salida
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        print("Tamaño de los logits:", logits.shape)

    else:
        logits=outputs.logits
    corresponde con una de las clases a segmentar.
    return logits


def create_masks(logits, prompts, threshold=0.2):
    """
    Crea máscaras sin superposición, calculando para cada una la probabilidad promedio de cada clase (simplificando cálculos futuros)
    
    Args:
    logits (torch.Tensor): Los logits del modelo. Shape [num_classes, 1, height, width].
    prompts (list of str): Etiquetas de clase correspondientes a los logits.
    threshold (float): Probabilidad mínima necesaria para que un píxel sea considerado como parte de una máscara.

    Salidas:
    dict: Un diccionario que contiene:
        'masks' (torch.Tensor): Un tensor de ints donde cada valor se corresponde con el índice de clase. Shape [height, width].
        'class_labels' (list): Lista de etiquetas de clase usadas en cada máscara.
        'mean_probabilities' (torch.Tensor): Un tensor que contiene las probabilidades promedio de cada clase para cada máscara. Shape [number of masks, number of classes].
    """
    # Convertimos los logits en probabilidades, usando una función sigmoide
    probabilities = torch.sigmoid(logits).squeeze(1)  # Shape pasa a ser [num_classes, height, width]

    # Asignamos el valor -1 para las regiones no clasificadas
    height, width = probabilities.shape[1], probabilities.shape[2]
    mask = torch.full((height, width), -1, dtype=torch.int)

    # Determinamos la clase más probable para cada píxel por encima del umbral definido
    max_probs, max_indices = torch.max(probabilities, dim=0)  # Shape [height, width] para ambos
    mask[max_probs > threshold] = max_indices[max_probs > threshold].type(torch.int)

    # Calculamos las probabilidades promedio para cada máscara
    num_classes = len(prompts)
    mean_probabilities = torch.zeros((num_classes, num_classes))
    
    for mask_index in range(num_classes):
        mask_area = (mask == mask_index)  # Máscara binaria para mask_index
        if mask_area.any(): 
            for class_index in range(num_classes):
                class_probs = probabilities[class_index]  # Mapa de probabilidades de la clase actual
                mean_probabilities[mask_index, class_index] = class_probs[mask_area].mean()
    
    
    return {
        'masks': mask, 
        'class_labels': prompts,
        'mean_probabilities': mean_probabilities 
    }

# Función para reescalar las máscaras al tamaño de las latentes
def downscale_mask_to_latent(masks, scale=(64,64)):

    # Añadir dimensiones de batch y channel
    masks = masks.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

    # Reducir la escala de las máscaras a las dimensiones (H,W) de las latentes
    downscaled_mask = F.interpolate(masks.float(), size=scale, mode='nearest')
    downscaled_mask=downscaled_mask.to(torch.int32)

    return downscaled_mask
