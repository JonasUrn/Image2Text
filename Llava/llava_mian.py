import requests
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import time


# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
def llava_get_caption(link, model, processor):
    image = Image.open(requests.get(link, stream=True).raw)
    prompt = "Describe the image in detail."
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        caption = processor.decode(generation, skip_special_tokens=True)
        return caption

def llava_main(links):
    print("google/paligemma-3b-pt-224")
    counter = 1
    start_time = time.time()
    model_id = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        revision="bfloat16",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    for image in links:
        caption = llava_get_caption(image, model, processor)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
