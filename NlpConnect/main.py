# Link to model: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning 

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO
import warnings
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 30
num_beams = 8
min_length = 15
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "min_length": min_length}

def get_images() -> list:
    images = []

    for i in range(0, 30):
        images.append(f"https://picsum.photos/id/{i}/200/300")
        
    return images

def predict_step(image_urls) -> list:
    images = []
    for image_url in image_urls:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        if img.mode != "RGB":
            img = img.convert(mode="RGB")

        images.append(img)
        
    pixel_values = feature_extractor(images=images, return_tensors="pt", padding=True).pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

urls = get_images()
counter = 0
for ans in predict_step(urls):
        print(f"Image {counter}: {ans}")
        counter += 1
