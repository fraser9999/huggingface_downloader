# Sample Inference Code for Picture Generating
# generated with Huggingface Model Downloader
# (Python Version 3.10.10 x64/amd64 with additional librarys)

import os
os.system('cls')

print('Importing Libs...please wait')
from diffusers import StableDiffusionPipeline
import torch
import random

model_path = r"runwayml/stable-diffusion-v1-5"

print('Model Inference')
print(str(model_path))

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
# alternative
# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

# gpu only rendering speed-up with next line
# pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')
# alternative
# pipe.to('cpu')

print('')
a=input('Enter Prompt>')
if a=='':
    prompt = 'a photo of an astronaut' 
else:
    prompt=str(a)

print('Prompt is: '+str(prompt))
print('')

while True:

    #Random Seed
    seed=random.randint(100,50000)
    print('Set Seed: ',str(seed))

    # use ...torch.Generator('cpu')... for CPU Render on next line
    g_cuda = torch.Generator('cuda').manual_seed(seed)

    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

    image.save('test.png')
    #<- Enter Code for saving/display multiple Pictures here
    image.show()

    a=input('Wait Key,ready...')

