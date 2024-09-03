import math
import torch
import numpy as np

import utils
import callbacks
import segmentators
import visualizers
import loaders


def prompt_prepare (prompts):
    if isinstance(prompts,list):
        prompt= " next to ".join(prompts)
    else: prompt=prompts
    return prompt

# Funciones para la adición de ruido sobre latentes

def compute_variance(p_1, p_other, k):
    variance=k * - math.log(p_1 *(1-p_other))
    print(f'varianza={variance}')
    return variance


def noising_latents(pipeline, latents, masks, mean_probabilities, betas, k=1, corr=0.0001, CFG_rescale=8):

    noisy_latents = latents.clone().detach().to('cuda')    

    for mask_value in np.unique(masks):
        for beta in reversed(betas):
            if mask_value == -1:
                mask_indices = torch.tensor((masks[:, 0, :, :] == mask_value)).to('cuda')        
                epsilon = torch.randn_like(noisy_latents)
            else:
                mask_indices = torch.tensor((masks[:, 0, :, :] == mask_value)).to('cuda')
                probabilities = mean_probabilities[mask_value]
                p_1 = probabilities[mask_value] + corr
                p_other = np.mean([p for idx, p in enumerate(probabilities) if idx != mask_value])+ corr
                variance = compute_variance(p_1, p_other, k)
                epsilon = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(variance)), size=noisy_latents.shape, device='cuda')
                
            noise = torch.sqrt(beta) * epsilon * CFG_rescale
            noisy_latents = torch.sqrt(1 - beta) * noisy_latents
          
            for c in range(noisy_latents.shape[1]):
                noisy_latents[:, c, :, :][mask_indices] = noisy_latents[:, c, :, :][mask_indices] + noise[:, c, :, :][mask_indices]

    return(noisy_latents)

# Función principal

def stable_tempering(pipeline,
                     prompts,
                     negative="Deformed, ugly, bad quality",
                     CFG=7.5,
                     CFG_rescale=8,
                     steps=20, 
                     stop_time=15, 
                     backsteps=5, 
                     k=4, 
                     seed=2024,
                     full_inference=False,
                     visualize_latents=True,
                     metrics=True,
                    ):
    
    prompt=prompt_prepare(prompts)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if full_inference==False:

        custom_callback = callbacks.CustomInterruptCallback(stop_time, pipeline.scheduler, pipeline.vae) #si .vae no funciona, extraer vae de model loader
    
        try:
            result = pipeline(
                            prompt=prompt,
                            negative_prompt=negative,
                            guidance_scale=CFG,
                            generator=generator,
                            num_inference_steps=steps,
                            callback=custom_callback,
                            callback_steps=1,
                            )
        except StopIteration:
            pass

        decoded_image = custom_callback.decoded_image
        decoded_image.show()
        utils.save_image(decoded_image)

        processor, segmentator_model=loaders.CLIPseg_loader() #TODO: sacar de esta función y llevar a otro lado para que sólo los cargue la primera vez, igual que los de SD

    
        logits = segmentators.segment_image(decoded_image, prompts, processor, segmentator_model)
        masks_dict = segmentators.create_masks(logits, prompts,threshold=0.1)

        resume_step = stop_time-backsteps
        latents = custom_callback.latents

        betas=custom_callback.recorded_betas[resume_step:]
        betas=torch.tensor(betas).to('cuda')
    
        visualizers.visualize_masks(masks_dict)
        
        masks = masks_dict["masks"]
        mean_probabilities = masks_dict["mean_probabilities"]
        decoded_tensor = custom_callback.decoded_tensor

        masks_scaled=segmentators.downscale_mask_to_latent(masks)

        noisy_latents=noising_latents(pipeline, latents, masks_scaled, mean_probabilities, betas, k=k,CFG_rescale=CFG_rescale)
        visualizers.visualize_latents(noisy_latents)

        resume_callback=callbacks.CustomResumeCallback(resume_step, pipeline.scheduler,noisy_latents)

        output = pipeline(
                        prompt=prompt,
                        image=None,
                        strength=100.0,
                        guidance_scale=CFG,
                        negative_prompt=negative,
                        generator=generator,
                        num_inference_steps=steps,
                        callback=resume_callback,
                        callback_steps=1        
                        )[0][0]
                       
        utils.save_image(output, prompt=prompt, seed=seed)

    else:
        print('NOTA: Durante la inferencia completa, se ignoran los parámetros stop_time, backsteps, k.')
        output = pipeline(
                        prompt=prompt,
                        image=None,
                        strength=100.0,
                        guidance_scale=CFG,
                        negative_prompt=negative,
                        generator=generator,
                        num_inference_steps=steps,
                        callback=None        
                        )[0][0]
        
        utils.save_image(output, prompt=prompt, seed=seed)

    return output
