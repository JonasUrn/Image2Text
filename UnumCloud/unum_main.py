from transformers import AutoModel, AutoProcessor, logging
from PIL import Image
import torch 
import requests
from io import BytesIO
import time
import transformers
from huggingface_hub import logging as hf_logging

hf_logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
logging.set_verbosity_error()

def get_images() -> list:
    images = []

    for i in range(0, 30):
        images.append(f"https://picsum.photos/id/{i}/200/300")
        
    return images

# Has to print true
# print(torch.cuda.is_bf16_supported())

def unum_get_caption(image_url, model, processor, device) -> str:
    prompt = "Describe the image in detail. Focus on the character and its features. Do not associate anyone with real-life personas, liek actors or profesionals athletes."
    response = requests.get(image_url, stream=True)
    image = Image.open(BytesIO(response.content))
    
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # inputs = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v.to(torch.long) for k, v in inputs.items()}
    
    # Need this to pipe all tensors to CUDA
    if torch.cuda.is_bf16_supported():
        model = model.to(device).to(torch.bfloat16)
        inputs = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v for k, v in inputs.items()}
    else:
        model = model.to(device)
        
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    caption = decoded_text[0:len(decoded_text) - 10]
    return caption

def unum_main(links):
    print("UNUM CLOUD MODEL")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: " + device)
    counter = 1
    start_time = time.time()
    model = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
    for image in links:
        caption = unum_get_caption(image, model, processor, device)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
    
    