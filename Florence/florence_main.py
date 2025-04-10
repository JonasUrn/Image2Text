# Link to original model: inputs = {k: v.to(device) for k, v in inputs.items()}

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, logging
import time
import transformers
from huggingface_hub import logging as hf_logging
import warnings

hf_logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
logging.set_verbosity_error()

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

def florence_generate_caption(link, processor, model, device, torch_dtype, text_input=None) -> str:
    image = Image.open(requests.get(link, stream=True).raw)
    task_prompt = "<MORE_DETAILED_CAPTION>"
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer["<MORE_DETAILED_CAPTION>"]

def florence_main(image_links):
    print("Microsoft Florence")
    start_time = time.time()
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: " + device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    counter = 1
    for link in image_links:
        caption = florence_generate_caption(link, processor, model, device, torch_dtype)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")

