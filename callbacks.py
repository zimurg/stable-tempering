## Este fichero contiene los callbacks que pasaremos al pipeline de Stablediffusion

import torch
from diffusers import SchedulerMixin, AutoencoderKL
from PIL import Image

import utils, visualizers

# Callback para interrumpir la inferencia en stop_time

class CustomInterruptCallback:
    def __init__(self, stop_time, scheduler, vae):
        """
        Se realizará una llamada a esta función en cada paso del proceso de inferencia del pipeline.
        
        Args:
        stop_time (int): El paso en el que se interrumpirá la inferencia.
        scheduler (diffusers.SchedulerMixin): El scheduler que se utilizará para programar el ruido en cada paso de inferencia.
        vae (diffusers.AutoencoderKL): Decodifica los latentes en un tensor con las dimensiones de la imagen final.
    
        Raises:
        StopIteration: Provoca una excepción, interrumpiendo la inferencia.
    
        Attr:
        .latents (torch.Tensor): Latentes en el paso stop_time.
        .decoded_image (PIL.Image): Imagen decodificada por el VAE en el paso stop_time.
        .recorded_betas (list): Contiene los valores de la varianza entre [0:stop_time].
        .noise_magnitude (list): magnitudes del ruido introducido entre [0:stop_time].

        Stores:
        decoded_latents (.png): guarda una imagen con el estado de las latentes en cada paso. TODO: variar el nombre con el seed para que no se sobreescriban
        """
        self.stop_time = stop_time
        self.scheduler = scheduler
        self.recorded_betas = []
        self.noise_magnitudes = []
        self.latents = None
        self.decoded_image = None
        self.decoded_tensor= None
        self.vae = vae.to("cuda")

    def __call__(self, step_idx, t, latents):

        beta = self.scheduler.betas[step_idx].item()
        self.recorded_betas.append(beta)
        noise = torch.randn_like(latents) * torch.sqrt(torch.tensor(beta)).to(latents.device)
        total_noise = noise.abs().mean().item()  # Debugging
        self.noise_magnitudes.append(total_noise)  # Debugging

        decoded_latents=visualizers.visualize_latents(latents)
        utils.save_latents(decoded_latents,step=step_idx, prompt="primera_pasada_")
        

        print(f"t= {step_idx}: Beta={beta}, Magnitud total del ruido={total_noise}, Valor mínimo de las latentes={latents.min().item()}, Valor máximo de las latentes={latents.max().item()}")

        if step_idx == self.stop_time:
            self.latents = latents

            self.decoded_tensor = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
            # TODO: convertir tensor en PIL.Image

                        
            self.decoded_image = utils.image_from_tensor(self.decoded_tensor)

            raise StopIteration


# Callback para reiniciar la inferencia en resume_step (stop_time-num_backsteps)

class CustomResumeCallback:
    def __init__(self, resume_step, scheduler,latents):

        """
        Se realizará una llamada a esta función en cada paso del proceso de inferencia del pipeline.
        
        Args:
        resume_step (int): El paso a partir de el cual se reiniciará la inferencia.
        scheduler (diffusers.SchedulerMixin): El scheduler que se utilizará para programar el ruido en cada paso de inferencia.
        latents (torch.Tensor): Las latentes que se introducirán como base para la inferencia
    
    
        Attr:
        .latents (torch.Tensor): Latentes en el paso stop_time.
        .decoded_image (PIL.Image): Imagen decodificada por el VAE en el paso stop_time.
        .recorded_betas (list): Contiene los valores de la varianza entre [0:stop_time].
        .noise_magnitude (list): magnitudes del ruido introducido entre [0:stop_time].  
    
        Stores:
        decoded_latents (.png): guarda una imagen con el estado de las latentes en cada paso. TODO: variar el nombre con el seed para que no se sobreescriban
        """
        
        self.resume_step = resume_step
        self.scheduler = scheduler
        self.noise_magnitudes = []
        self.latents = latents

    def __call__(self, step_idx, t, latents):
            
        beta = self.scheduler.betas[step_idx].item()
        noise = torch.randn_like(latents) * torch.sqrt(torch.tensor(beta)).to(latents.device) 
        total_noise = noise.abs().mean().item()  # Debugging
        self.noise_magnitudes.append(total_noise)  # Debugging

        print(f"t= {step_idx}: Beta={beta}, Magnitud total del ruido={total_noise}, Valor mínimo de las latentes={latents.min().item()}, Valor máximo de las latentes={latents.max().item()}")


        if step_idx < self.resume_step:
            latents.copy_(self.latents)  #TODO: en t=0, el modelo parte de latentes vacías, así que si resume_step es 0, sería una inferencia normal y corriente

        else: 
            decoded_latents=visualizers.visualize_latents(latents)
            utils.save_latents(decoded_latents,step=step_idx, prompt="segunda_pasada_")
