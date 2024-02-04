from fastapi import FastAPI
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

save_directory = "generated_images"
os.makedirs(save_directory, exist_ok=True)

#heartbeat
@app.get("/heartbeat")
async def root():
    return {"status": "alive"}

@app.post("/generate_image")
async def generate_image(prompt, width, height, guidance_scale, negative_prompt):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to("mps")
    image = pipe(prompt, width=int(width), height=int(height), guidance_scale=float(guidance_scale), negative_prompt=negative_prompt, num_inference_steps=5).images[0]
    print(image)
    image_path = os.path.join(save_directory, "generated_image.png")
    image.save(image_path)

    return {"status": "success", "image_path": image_path}