from transformers import AutoModel, AutoProcessor, logging
from PIL import Image
import torch 
import requests
from io import BytesIO

def get_images() -> list:
    images = []

    for i in range(0, 30):
        images.append(f"https://picsum.photos/id/{i}/200/300")
        
    return images

# Has to print true
print(torch.cuda.is_bf16_supported())

model = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)


def get_caption(image_url, model, processor):
    prompt = "Describe the image"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    model = model.to(torch.bfloat16)
    inputs = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v.to(torch.long) for k, v in inputs.items()}
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

    return decoded_text[0:len(decoded_text) - 10]

c = 0
for url in get_images():
    caption = get_caption(url, model, processor)
    print(f"Image {c}: {caption}")
    c += 1