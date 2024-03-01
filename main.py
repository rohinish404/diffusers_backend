from fastapi import FastAPI, HTTPException
import torch
from diffusers import StableDiffusionPipeline,DDPMScheduler,UNet2DModel
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tqdm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

save_directory = "generated_images"
os.makedirs(save_directory, exist_ok=True)

class ImageGenerationRequest(BaseModel):
    prompt: str
    width: int
    height: int
    guidanceScale: float
    negativePrompt: str
    
repo_id = "google/ddpm-church-256"

#heartbeat
@app.get("/heartbeat")
async def root():
    return {"status": "alive"}

@app.post("/generate_image")
async def generate_image(request_data: ImageGenerationRequest):
    prompt = request_data.prompt
    width = request_data.width
    height = request_data.height
    guidance_scale = request_data.guidanceScale
    negative_prompt = request_data.negativePrompt
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("mps")
        image = pipe(prompt, width=int(width), height=int(height), guidance_scale=float(guidance_scale), negative_prompt=negative_prompt, num_inference_steps=5).images[0]
        # image_path = os.path.join(save_directory, "generated_image.png")
        # image.save(image_path)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return {"status": "success", "image_path": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
    
    
@app.post("/custom_denoising")   
async def custom_denoising():
    #load scheduler and model
    scheduler = DDPMScheduler.from_config(repo_id)
    model = UNet2DModel.from_pretrained(repo_id)
    #set timesteps
    scheduler.set_timesteps(num_inference_steps=15)

    noisy_sample = torch.randn(
        1, 3, model.config.sample_size, model.config.sample_size
    )
    sample = noisy_sample

    for t in tqdm.tqdm(scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t).sample

        
        previous_noisy_sample = scheduler.step(residual, t, sample).prev_sample
        sample = previous_noisy_sample
         
    image = (sample / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    
    image_path = os.path.join(save_directory, "generated_image.png")
    image.save(image_path)
    return {"status": "success", "image_path": image_path}   
    
    