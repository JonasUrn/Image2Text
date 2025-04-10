# Link to original model: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time

def meta_get_caption(link, model, processor) -> str:
    image = Image.open(requests.get(link, stream=True).raw)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    
    return processor.decode(output[0])

def meta_main(image_links):
    print("META LLAMA - Llama-3.2-11B-Vision-Instruct")
    start_time = time.time()
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    counter = 1
    for link in image_links:
        caption = meta_get_caption(link, model, processor)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")