# With quantization

import torch
import numpy as np
from tqdm import tqdm
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
from io import BytesIO

device = torch.device("cuda")

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, load_in_8bit=True)

model.eval()

# Clear any previously allocated memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats("cuda:0")

text_prompt = "The character is at the top left in the orange square with an S. The light blue squares with an F are places where you can move to. The dark blue squares with an H are holes that you should avoid. The green square with a G is your goal. Where should the character move to go towards the goal? Choose an action from down, left, or right. The character should move"

# Fetch and prepare the image
url = "https://149695847.v2.pressablecdn.com/wp-content/uploads/2018/03/Frozen-Lake.png"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open the image from bytes
    image = Image.open(BytesIO(response.content))
else:
    print(f"Failed to retrieve the image. Status code: {response.status_code}")
    image = None  # Make sure to handle the case where the image is not retrieved

# Tokenize the prompt
inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

# Get the length of the tokenized prompt
prompt_length = inputs.input_ids.size(1)

# Run inference
try:
    with torch.no_grad():  # Disable gradient tracking
        # Generate tokens
        generation_output = model.generate(**inputs, max_new_tokens=5)

        # Slice the generation_output to exclude the prompt
        newly_generated_tokens = generation_output[:, prompt_length:]

        # Decode the newly generated tokens
        generation_text = processor.batch_decode(newly_generated_tokens, skip_special_tokens=True)

        # Display the generated text
        print(generation_text)

except RuntimeError as e:
    print(e)

# Print memory usage
print(f"Allocated memory: {torch.cuda.memory_allocated('cuda:0') / 1e9:.2f} GB")
print(f"Cached memory: {torch.cuda.memory_reserved('cuda:0') / 1e9:.2f} GB")
print(f"Peak allocated memory: {torch.cuda.max_memory_allocated('cuda:0') / 1e9:.2f} GB")