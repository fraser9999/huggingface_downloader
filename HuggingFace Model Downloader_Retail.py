# Hermanns Huggingface/Diffusers Model Downloader  
# 30.7.2023 Version 0.1a
#
# Contact:hermann.knopp@gmx.at
# Hermann Knopp 2023 
#
# please install Python 3.10.10 (x64) and Librarys
# open command box , after installing python
# use: pip install -r requirements.txt  
# perhaps you should make virtual environment
# with venv if you have other python setups  
#
# for Inference: (generating pictures) 
# this version only supports NVIDIA Cuda GFX-Cards
# minimum requirements GForce GTX 750 with 2/4GB VRam
# use FP16 Model with 2GB or FP32 Model with 4GB
# for performance: Nvidia RTX 3060/12GB or better... 
# 
# but: can be adapted for slow CPU Rendering also
# (Device="cpu" with Diffuser/Torch CPU Librarys)
# look at: requirements-cpu.txt
#
# tested on Nvidia RTX 3060/12GB


# OS Librarys
import os
import os.path


# Imports Libs
import time
import random
from datetime import datetime


# Title 
os.system("title Hermanns Huggingface/Diffusers Binary Model Downloader")
os.system('mode con: cols=130 lines=50')
os.system("cls")
print("Resize Window/Importing Libs...please wait")


# Diffuser/Torch Librarys
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


#Set Working Dir
dir_path = os.path.dirname(os.path.realpath(__file__))


# Main App Text Message
print("Hermanns Huggingface/Diffuser Binary Model Downloader")
print("")
print("Local Model Save Directory is: c:\\users\\username\\.cache")
print("Please use Symlinks if you would change Directory Location...")
print("")
print("")


# Wait for User
a=input("Wait Key..")
#os.system("cls")
print("")


# Prompt Huggingface.co  Path Variable Input Box
print("Please insert Huggingface.co Model Path")
print("")
print("for Sample Path: 'runwayml/stable-diffusion-v1-5' press Enter ")
print("")


# Input Box
mp=input("Path> ")


# Check Nonsense
if mp==None or mp=="":
   model_path="runwayml/stable-diffusion-v1-5"
else:
   model_path=mp



# Wait for User
print("")
a=input("Wait Key..ready to download")
print("")


# Status Message
print("Seting up Dummy Diffuser Pipeline for Download only.")
print("Downloading some 2/4GB Files..please wait, will take some Time")
print("")


# Main Download Routine, for Setting up Diffuser Pipeline...
# Dummy Setup for Download only

try:
   # First Try FP32 Download
   print("Trying to download FP32 Version first...")
   print("")
   scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
   pipe = StableDiffusionPipeline.from_pretrained(model_path,revision="fp32", scheduler=scheduler, safety_checker=None,torch_dtype=torch.float32)

except:
   # Second Try FP16 Download
   print("Error at downloading FP32 Version.. switch to Standard/FP16 Version...")
   print("") 
   scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
   pipe = StableDiffusionPipeline.from_pretrained(model_path,scheduler=scheduler, safety_checker=None,torch_dtype=torch.float32)


# Display Message
print("")
print("Download Finished!")
print("")


# Display Sourcecode for Picture Inference
print("For Inference use this Code:(will be saved too)")
print("Copy with crtl+c , paste with crtl+v ")
print("")
print("")

# Wait User
a=input("Wait key...")
os.system("cls")

# Sourcecode Preview
print("Code Start>------------------------------------------------------------------------------")
print("import os")
print("os.system('cls')")
print("print('Importing Libs...please wait')")
print("from diffusers import StableDiffusionPipeline")
print("import torch")
print("import random")
print("model_path = r"+chr(34)+str(model_path)+chr(34))
print("")
print("print('Model Inference')")
print("print(str(model_path))")
print("pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)")
print("# alternative")
print("# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)")
print("pipe.to('cuda')")
print("# alternative")
print("# pipe.enable_xformers_memory_efficient_attention()")
print("# pipe.to('cpu')")
print("print('')")
print("a=input('Enter Prompt>')")
print("if a=='':")
print("    prompt = 'a photo of an astronaut' ")
print("else:")
print("    prompt=str(a)")
print("print('Prompt is: '+str(prompt))")
print("print('')")
print("")
print("while True:")
print("    #Random Seed")
print("    seed=random.randint(100,50000)")
print("    print('Set Seed: ',str(seed))")
print("    g_cuda = torch.Generator('cuda').manual_seed(seed)")
print("")
print("    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]")
print("    image.save('test.png')")
print("    image.show()")
print("    a=input('Wait Key,ready...')")
print("Code End>--------------------------------------------------------------------------------")
print("")


# Wait User
a=input("Wait key...")


# Main Save Code Routine

# Get the current working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
print("Work Dir is:", dir_path)


# Datetime object containing current date and time for multiple Filenames
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
filename = dir_path + "\\" + "Inference_Code_" + dt_string + ".py"
print("Filename is: ",filename)


# Open File for Sourcecode writing
f = open(filename, "a")


# Write Python Inference SourceCode to Text File
f.write("# Sample Inference Code for Picture Generating")
f.write("\n")
f.write("# generated with Huggingface Model Downloader")
f.write("\n")
f.write("# (Python Version 3.10.10 x64/amd64 with additional librarys)")
f.write("\n")
f.write("\n")
f.write("import os")
f.write("\n")
f.write("os.system('cls')")
f.write("\n")
f.write("")
f.write("\n")
f.write("print('Importing Libs...please wait')")
f.write("\n")
f.write("from diffusers import StableDiffusionPipeline")
f.write("\n")
f.write("import torch")
f.write("\n")
f.write("import random")
f.write("\n")
f.write("")
f.write("\n")
f.write("model_path = r"+chr(34)+str(model_path)+chr(34))
f.write("\n")
f.write("")
f.write("\n")
f.write("print('Model Inference')")
f.write("\n")
f.write("print(str(model_path))")
f.write("\n")
f.write("")
f.write("\n")
f.write("pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)")
f.write("\n")
f.write("# alternative")
f.write("\n")
f.write("# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)")
f.write("\n")
f.write("")
f.write("\n")
f.write("# gpu only rendering speed-up with next line")
f.write("\n")
f.write("# pipe.enable_xformers_memory_efficient_attention()")
f.write("")
f.write("\n")
f.write("pipe.to('cuda')")
f.write("\n")
f.write("# alternative")
f.write("\n")
f.write("# pipe.to('cpu')")
f.write("\n")
f.write("")
f.write("\n")
f.write("print('')")
f.write("\n")
f.write("a=input('Enter Prompt>')")
f.write("\n")
f.write("if a=='':")
f.write("\n")
f.write("    prompt = 'a photo of an astronaut' ")
f.write("\n")
f.write("else:")
f.write("\n")
f.write("    prompt=str(a)")
f.write("\n")
f.write("")
f.write("\n")
f.write("print('Prompt is: '+str(prompt))")
f.write("\n")
f.write("print('')")
f.write("\n")
f.write("")
f.write("\n")
f.write("while True:")
f.write("\n")
f.write("")
f.write("\n")
f.write("    #Random Seed")
f.write("\n")
f.write("    seed=random.randint(100,50000)")
f.write("\n")
f.write("    print('Set Seed: ',str(seed))")
f.write("\n")
f.write("\n")
f.write("    # use ...torch.Generator('cpu')... for CPU Render on next line")
f.write("\n")
f.write("    g_cuda = torch.Generator('cuda').manual_seed(seed)")
f.write("\n")
f.write("")
f.write("\n")
f.write("    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]")
f.write("\n")
f.write("")
f.write("\n")
f.write("    image.save('test.png')")
f.write("\n")
f.write("    #<- Enter Code for saving/display multiple Pictures here")
f.write("\n")
f.write("    image.show()")
f.write("\n")
f.write("")
f.write("\n")
f.write("    a=input('Wait Key,ready...')")
f.write("\n")
f.write("")
f.write("\n")


#Close Text File
f.close()


#End Message
print("File Saved.to Disk.")
print("")
print("Use/Start Inference_Code_xxxx.py for Generating Pictures ")
print("Code is already working with your downloaded .cache Model.")
a=input("Wait Key, to End")
quit()

