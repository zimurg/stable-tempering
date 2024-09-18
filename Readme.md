# Tempering Stable Diffusion

Este proyecto implementa un proceso de inferencia mejorado para despliegues de modelos de la familia *Stable Diffusion*, basado en técnicas de "simulated annealing" para reducir el "sangrado conceptual" en imágenes generadas con múltiples sujetos.

## Descripción

El objetivo de este prototipo experimental es mejorar la capacidad de los modelos de la familia *Stable Diffusion* para generar imágenes coherentes con múltiples sujetos sin que sus características se mezclen. Para lograrlo, se utiliza un proceso inspirado en el templado metalúrgico, en el que se deshacen pasos de inferencia mediante la inyección de ruido controlado durante el proceso de inferencia en base a una segmentación de las latentes, permitiendo que el modelo escape de mínimos locales persistentes.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/usuario/tempering-stable-diffusion.git
   ```

2. No es necesario instalar las dependencias, ya que lo harán automáticamente al inicializar init.py

3. Si se producen problemas con las librerías instaladas, ejecutar:
    ```bash
    python init.py --update
    ```
## uso

Existen 3 flujos de trabajo que se pueden inicializar desde init.py, mediante sus argumentos:

    • [--update]: Actualiza las librerías según “requirements.txt”.
    • [--tempered_inference]: Realiza una pasada de inferencia temperada cuyos parámetros pueden ser introducidos como argumentos.
    • [--evaluate]: Realiza [–cycles] tests en los que compara una inferencia normal con el modelo contra una inferencia temperada, obteniendo las métricas PPV (precisión), NPV y Distancia Jensen-Shannon, así como presentando una gráfica en la que se comparan las dos primeras como nube de puntos.

A estos flujos se pueden agregar además los siguientes argumentos:

    • [--prompts]: Los prompts a introducir en los modelos; cada sujeto debe ser introducido entre comillas, independientemente.
    • [--steps]: El número de pasos de inferencia que realizarán los modelos.
    • [--stoptime]: El paso en que se interrumpirá la inferencia.
    • [--backsteps]: El número de pasos de ruido que se introducirán después de stoptime.
    • [--cfg_rescale]: El valor de reescalado del ruido para compensar el impacto de la CFG.
    • [--k]: El impacto que la incertidumbre entre las clases tiene sobre el ruido introducido.
    • [--seed]: la semilla inicial.
    • [--cycles]: número de experimentos realizados por [–evaluate]


En la primera pasada de [--tempered_inference], o [--evaluate], además se instalarán desde el repositorio de *Huggingface* los modelos definidos en loaders.py, que comprenden la U-net, el VAE y el scheduler de *Stable Diffusion*, el modelo de lenguaje y el tokenizador de CLIP, así como el modelo de texto y el procesador de CLIPseg. También se cargará el pipeline de Text2Img de la librería *Diffusers*.

## Ejemplos

### Inferencia temperada

   ```bash
   python main.py --tempered_inference --prompts "a black dog" "a toy robot" --seed 2024 --steps 20 --stoptime 15 --backsteps 5 --cfg_rescale 10
   ```

### Tests de evaluación

   ```bash
   python main.py --evaluate --prompts "a black dog" "a toy robot" --seed 2024 --steps 20 --stoptime 15 --backsteps 5 --cfg_rescale 10 --cycles 4
   ```

## Créditos

Este proyecto fue desarrollado por Alberto García Núñez como parte del Trabajo Fin de Estudios en el Máster Universitario en Inteligencia Artificial de la Universidad Internacional de La Rioja, dirigido por Abdelmalik Moujahid Moujahid

En este prototipo se ha incorporado código procedente de las siguientes fuentes, respetando las licencias de uso pertinentes:

   @Librería Diffusers (von Platen et al 2022), disponible [aquí](https://github.com/huggingface/diffusers)
   @Función de escalado de latentes sin VAE (Vass, 2024), disponible [aquí](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space)
   @Tutorial de segmentación *Zero-Shot* con CLIPseg (Rogge, 2022), disponible [aquí](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb)

Los modelos empleados en el despliegue por defecto se corresponden con:

   @CompVis/stable-diffusion-v1-4 (Rombach, et al,2022), disponible [aquí](https://huggingface.co/CompVis/stable-diffusion-v1-4)
   @CIDAS/clipseg-rd64-refined (CIDAS, 2022), disponible [aquí](https://huggingface.co/CIDAS/clipseg-rd64-refined)

## Disclaimer

Los modelos y el software utilizados en este proyecto están sujetos a los derechos de propiedad intelectual de sus respectivos creadores. Cualquier infracción no intencionada será rectificada tan pronto se detecte.

El autor no se hace responsable de prácticas no éticas realizadas por los creadores de los modelos para su entrenamiento. Tampoco se hace responsable del uso indebido que terceras partes hagan del código proporcionado.

