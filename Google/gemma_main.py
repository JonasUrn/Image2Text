# Link to original source: https://huggingface.co/google/gemma-3-27b-it

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, pipeline
from PIL import Image
import requests
import torch
import time

# def gemma_get_caption(link, model, processor) -> str:
#     messages = [
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": "You are a system that describes images."}]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": link},
#                 {"type": "text", "text": "Describe this image in detail."}
#             ]
#         }
#     ]
    
#     inputs = processor.apply_chat_template(
#         messages, 
#         add_generation_prompt=True, 
#         tokenize=True,
#         return_dict=True, 
#         return_tensors="pt"
#     ).to(model.device, dtype=torch.bfloat16)

#     input_len = inputs["input_ids"].shape[-1]

#     with torch.inference_mode():
#         generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#         generation = generation[0][input_len:]

#     caption = processor.decode(generation, skip_special_tokens=True)
#     return caption

def gemma_get_caption(link, device):
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-27b-it",
        device="cuda",
        torch_dtype=torch.bfloat16
    )
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a system that describes messages."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": link},
                {"type": "text", "text": "Describe the image in detail."}
            ]
        }
    ]

    output = pipe(text=messages, max_new_tokens=200)
    return output[0]["generated_text"][-1]["content"]

def gemma_main(image_links):
    print("Google - gemma-3-27b-it")
    start_time = time.time()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: " + device)
    counter = 1
    
    for link in image_links:
        caption = gemma_get_caption(link, device)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")