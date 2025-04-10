# Link to implementation source: https://huggingface.co/Salesforce/blip-image-captioning-large

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import time
    
def salesforce_get_caption(link, model, processor) -> str:
    raw_image = Image.open(requests.get(link, stream=True).raw).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption
    

def salesforce_main(image_links):
    print("Salesforce - blip-image-captioning-large")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: " + device)
    start_time = time.time()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    counter = 1
    for link in image_links:
        caption = salesforce_get_caption(link, model, processor)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
