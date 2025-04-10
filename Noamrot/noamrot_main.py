# Link to original source: https://huggingface.co/noamrot/FuseCap_Image_Captioning
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import time

def noamrot_get_caption(link, model, processor):
    image = Image.open(requests.get(link, stream=True).raw)
    
    inputs = processor(image, "Picture", return_tensors='pt').to(model.device)
    
    out = model.generate(**inputs, 
        num_beams=5,
        max_new_tokens=100,  
        early_stopping=True,       
        no_repeat_ngram_size=2,     
        length_penalty=1.0   
        )
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption
    

def noamrot_main(links):
    print("noamrot/FuseCap_Image_Captioning")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: " + device)
    counter = 1
    start_time = time.time()
    processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
    model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)
    for image in links:
        caption = noamrot_get_caption(image, model, processor)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
