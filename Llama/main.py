from openai import OpenAI

client = OpenAI(
    base_url="https://api.thehive.ai/api/v3/",
    api_key="<API_KEY>" 
)

def get_completion(prompt, image_link, model = "meta-llama/llama-3.2-11b-vision-instruct"):
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

counter = 0
for image in get_images():
    print(f'Image {counter}: {get_completion("Describe the image. Keep the description under 30 words.", image)}')
    counter += 1
