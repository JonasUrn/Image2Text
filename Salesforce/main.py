# Link to implementation source: https://huggingface.co/Salesforce/blip-image-captioning-large

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from image_handling import get_images
import torch

print(f"Cuda available: {torch.cuda.is_available()}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

counter = 0
for image in get_images():

    img_url = image
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    print(f"Image {counter}: {processor.decode(out[0], skip_special_tokens=True)}")
    counter += 1

