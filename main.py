from fastapi import FastAPI, HTTPException
import torch
from diffusers import StableDiffusionPipeline
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO


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