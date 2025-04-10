#Link to original source: https://huggingface.co/bipin/image-caption-generator
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import time
import requests

def bipin_get_caption(link, model, feature_extractor, tokenizer, device):
    img = Image.open(requests.get(link, stream=True).raw)
    if img.mode != 'RGB':
        img = img.convert(mode="RGB")
    
    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    max_length = 500
    num_beams = 4

    output_ids = model.generate(pixel_values, num_beams=num_beams, max_length=max_length)

    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return preds

def bipin_main(links):
    print("bipin/image-caption-generator")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    counter = 1
    start_time = time.time()
    model_name = "bipin/image-caption-generator"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for image in links:
        caption = bipin_get_caption(image, model, feature_extractor, tokenizer, device)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
