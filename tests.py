import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy.spatial.distance import jensenshannon

import loaders
import segmentators
import stable_tempering
import utils

def calculate_precision_npv(mean_probabilities, prompts):
    precisions = []
    npvs = []

    for class_index in range(len(prompts)):
        # Probabilidad promedio de la clase actual (verdaderos positivos)
        tp = mean_probabilities[class_index, class_index].item()

        # Probabilidad promedio de las clases secundarias (falsos positivos)
        fp = mean_probabilities[class_index, :].sum().item() - tp

        # True negatives: sum of probabilities of other classes being absent
        tn = 0
        for i in range(len(prompts)):
            if i != class_index:
                tn += (1 - mean_probabilities[i, class_index].item())

        # False negatives: probability mass assigned to other classes for the true class area
        fn = mean_probabilities[:, class_index].sum().item() - tp

        # Calculate precision (PPV)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)

        # Calculate NPV in a similar manner
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npvs.append(npv)

    avg_precision = np.mean(precisions)
    avg_npv = np.mean(npvs)

    return avg_precision, avg_npv

def run_inference_test(prompts, seed=2024, steps=20, stop_time=15, backsteps=5, k=4, CFG_rescale=10, cycles=1, threshold=0.1):
    pipeline = loaders.SD_loader()
    processor, segmentator_model = loaders.CLIPseg_loader()

    jsd_normal = []
    jsd_tempered = []

    precision_normal = []
    precision_tempered = []

    npv_normal = []
    npv_tempered = []

    prompt=stable_tempering.prompt_prepare(prompts)

    for n in range(cycles):
        seed = seed + n * 9973  # Variabilidad en la semilla de forma determinista

        # Inferencia normal
        generator = torch.Generator(device="cuda").manual_seed(seed)
        normal_image = pipeline(prompt=prompt,
                                generator=generator,
                                num_inference_steps=steps).images[0]

        utils.save_image(normal_image, prompt="normal_gen_"+prompt, seed=seed)

        # Inferencia temperada
        tempered_image = stable_tempering.stable_tempering(
            pipeline=pipeline,
            prompts=prompts,
            steps=steps,
            stop_time=stop_time,
            backsteps=backsteps,
            k=k,
            CFG_rescale=CFG_rescale,
            seed=seed
        )

        # Segmentación
        logits_normal = segmentators.segment_image(normal_image, prompts, processor, segmentator_model)
        logits_tempered = segmentators.segment_image(tempered_image, prompts, processor, segmentator_model)

        masks_dict_normal = segmentators.create_masks(logits_normal, prompts, threshold=threshold)
        masks_dict_tempered = segmentators.create_masks(logits_tempered, prompts, threshold=threshold)

        # Cálculo de las métricas
        mean_probs_normal = masks_dict_normal['mean_probabilities']
        mean_probs_tempered = masks_dict_tempered['mean_probabilities']

        for i in range(len(prompts)):

            p_true = np.zeros(len(prompts))
            p_true[i] = 1

            p_normal = np.clip(mean_probs_normal[i].cpu().numpy(), 1e-8, 1)
            p_tempered = np.clip(mean_probs_tempered[i].cpu().numpy(), 1e-8, 1)

            if np.any(p_normal > 0) and np.any(p_tempered > 0):
                jsd_normal.append(jensenshannon(p_true, p_normal) ** 2)
                jsd_tempered.append(jensenshannon(p_true, p_tempered) ** 2)
            else:
                jsd_normal.append(0)
                jsd_tempered.append(0)

        # Llamada a la función modificada con probabilidades precalculadas
        precision_n, npv_n = calculate_precision_npv(mean_probs_normal, prompts)
        precision_t, npv_t = calculate_precision_npv(mean_probs_tempered, prompts)

        precision_normal.append(precision_n)
        npv_normal.append(npv_n)

        precision_tempered.append(precision_t)
        npv_tempered.append(npv_t)

    avg_jsd_normal = np.mean(jsd_normal)
    avg_jsd_tempered = np.mean(jsd_tempered)

    avg_precision_normal = np.mean(precision_normal)
    avg_npv_normal = np.mean(npv_normal)

    avg_precision_tempered = np.mean(precision_tempered)
    avg_npv_tempered = np.mean(npv_tempered)

    # Representación de los resultados
    results = pd.DataFrame({
        'Métrica': ['JSD', 'Precisión', 'NPV'],
        'Normal': [avg_jsd_normal, avg_precision_normal, avg_npv_normal],
        'Temperada': [avg_jsd_tempered, avg_precision_tempered, avg_npv_tempered]
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results.values, colLabels=results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title(f'Métricas Comparativas: Normal vs Temperada\n(Prompt: {prompt})')
    plt.savefig(f"Metricas_{prompt}.png")
    plt.show()

    # Curva PPV/NPV
    plt.figure(figsize=(10, 8))
    plt.scatter(npv_normal, precision_normal, color='blue', label='Inferencia normal')
    plt.scatter(npv_tempered, precision_tempered, color='red', label='Inferencia temperada')
    plt.xlabel("NPV (Negative Predictive Value)")
    plt.ylabel("PPV (Precisión)")
    plt.title("Curva PPV / NPV")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"PPVoNPV_graph_{prompt}.png")
    plt.show()

    return {
        "JSD_normal": avg_jsd_normal,
        "JSD_tempered": avg_jsd_tempered,
        "Precision_normal": avg_precision_normal,
        "NPV_normal": avg_npv_normal,
        "Precision_tempered": avg_precision_tempered,
        "NPV_tempered": avg_npv_tempered
    }
