# Diffusers Project
I'll be using this readme to document stuff basically. Maybe something like my thinking around this project, what i need to get done, what i should focus on, etc etc..
## 24th Feb, 2024
I pushed the changes earlier but haven't gotten time to add new things to project or edit the readme. So, what's the current state of progress and project? Firstly, the first commit i did was to just display an image using a stable-diffusion-v1-5 model by giving it a prompt and various other parameters such as height, width, guidance_scale,  negative_prompt and num_inference_steps. Let me explain a bit of how i did this (actually took help of diffusers starter colab) and a bit about parameters also.
So, diffusers lib present on hugging face provides various pipelines to load pretrained models which can then be used for image generation. There are many other tasks as well such as image to image, inpainting, etc. (i don't know about most of them) but firstly we'll be looking into text to generation only. The pipeline that i'm using for this is a StableDiffusionPipeline which will load the above mentioned model. Now, coming to parameters - 
prompt - The description of the image that you want to generate.
Width, height - The height and width of generated image (setting it to 50 50 both because i dont have a gpu so larger height and widht with take much longer time).
guidance_scale - How much the prompt should influence the image generation. Less means more creativity
negative_prompt - 
num_inference_steps - 


