from PIL import Image
from tqdm.auto import tqdm
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler

class StableDiffusionPipeline:
    def __init__(self, model_name, seed, device = "cpu"):
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_safetensors=True)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", use_safetensors=True
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", use_safetensors=True
        )
        self.scheduler = UniPCMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.seed = seed

        self.torch_device = device
        self.vae.to(self.torch_device)
        self.text_encoder.to(self.torch_device)
        self.unet.to(self.torch_device)
        self.generator = torch.manual_seed(self.seed) 
    
    def generate_embeddings(self,prompt, batch_size):
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
            
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]


        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])   
        return text_embeddings 
    
    def get_latents(self, batch_size, height, width):
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=self.generator,
            device=self.torch_device,
        )
        return latents
    
    def generate_image(self, prompt, height, width, num_inference_steps, guidance_scale):
            
        batch_size = len(prompt)
                
        text_embeddings = self.generate_embeddings(prompt, batch_size)

        latents = self.get_latents(batch_size, height, width)
        
        latents = latents * self.scheduler.init_noise_sigma
        
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        return image    
            
       
                    
                                
            




# prompt = ["a photograph of an astronaut riding a horse"]
# height = 512  # default height of Stable Diffusion
# width = 512  # default width of Stable Diffusion
# num_inference_steps = 25  # Number of denoising steps
# guidance_scale = 7.5  # Scale for classifier-free guidance
# generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise



#         image = (image / 2 + 0.5).clamp(0, 1).squeeze()
#         image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
#         images = (image * 255).round().astype("uint8")
#         image = Image.fromarray(image)
#         image            
                
