from openai import OpenAI
import os
import time

def llama_get_caption(image_link, client) -> str:
    prompt = "Describe the image."
    model = "meta-llama/llama-3.2-11b-vision-instruct"
    response = client.chat.completions.create(
      model=model,
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
            {
              "type": "image_url",
              "image_url": {
              "url": image_link
              }
            }
          ]
        }
      ],
      temperature=0.7,
      max_tokens=1000
    )

    return response.choices[0].message.content

def get_images() -> list:
    images = []

    for i in range(0, 30):
        images.append(f"https://picsum.photos/id/{i}/200/300")
        
    return images
  
def llama_main(image_links):
    print("HIVE API - llama-3.2-11b-vision-instruct")
    API_KEY = os.environ.get("HIVE_API_KEY", "ERROR")
    start_time = time.time()
    client = OpenAI(
        base_url="https://api.thehive.ai/api/v3/",
        api_key="PY7s9DzEhLUMF8MsdyI5XZ3yJuuYgkMW"
    )
    counter = 1
    for link in image_links:
        caption = llama_get_caption(link, client)
        print(f"Picture No.{counter}: " + caption)
        counter += 1
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.6f} seconds\n\n\n")
