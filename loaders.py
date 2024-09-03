import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


def SD_loader(model ="CompVis/stable-diffusion-v1-4",
                 checkpoint = "CompVis/stable-diffusion-v1-4",
                 CLIP = "CompVis/stable-diffusion-v1-4",
                 vae = "stabilityai/sd-vae-ft-mse"):


    vae = AutoencoderKL.from_pretrained(vae)
    text_encoder = CLIPTextModel.from_pretrained(CLIP, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")
    scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(model, subfolder="feature_extractor")

    pipeline_args={"vae": vae, "text_encoder": text_encoder, "tokenizer": tokenizer, "unet": unet, "scheduler": scheduler, "safety_checker": None, "feature_extractor": feature_extractor}
    pipeline = StableDiffusionPipeline(**pipeline_args).to("cuda")

    return pipeline

def CLIPseg_loader(model="CIDAS/clipseg-rd64-refined"):

    processor = CLIPSegProcessor.from_pretrained(model)
    model = CLIPSegForImageSegmentation.from_pretrained(model)

    return processor, model

