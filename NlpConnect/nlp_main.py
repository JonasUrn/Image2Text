# Link to model: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning 

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, logging
import torch
from PIL import Image
import requests
from io import BytesIO
import warnings
import transformers
import time
from huggingface_hub import logging as hf_logging

hf_logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
logging.set_verbosity_error()

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

def get_images() -> list:
    images = []

    for i in range(0, 30):
        images.append(f"https://picsum.photos/id/{i}/200/300")
        
    return images

def nlp_connect_get_caption(image_url, device, feature_extractor, model, tokenizer, gen_kwargs) -> str:
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    if img.mode != "RGB":
        img = img.convert(mode="RGB")
        
    pixel_values = feature_extractor(images=[img], return_tensors="pt", padding=True).pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def nlp_connect_main(image_links):
    print("NLP Connect")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: " + device)
    start_time = time.time()
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.to(device)
    max_length = 30
    num_beams = 8
    min_length = 15
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "min_length": min_length}
    counter = 1
    for link in image_links:
        caption = nlp_connect_get_caption(link, device, feature_extractor, model, tokenizer, gen_kwargs)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
