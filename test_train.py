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

# Clear any previously allocated memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats("cuda:0")

import random
import string
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, FuyuForCausalLM, FuyuProcessor

# Generate random text
def generate_random_text(length=100):
    """ Generate a random string of letters """
    letters = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choice(letters) for i in range(length))

# Parameters
num_examples = 100
max_length = 128
batch_size = 1

# Prepare the dataset
texts = [generate_random_text(max_length) for _ in range(num_examples)]
data = {'text': texts}
dataset = Dataset.from_dict(data)

# Tokenize the dataset
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")

def tokenize_function(examples):
    return processor(examples["text"], truncation=True, max_length=max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model
model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", load_in_4bit=True)

# LoRA configuration
lora_config = LoraConfig(
    target_modules=["query_key_value"],
    init_lora_weights=False
)
model.add_adapter(lora_config, adapter_name="lora")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Custom data collator
class CustomDataCollator:
    def __call__(self, features):
        # Convert list of features into a batch of tensors
        batch = {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}
        return batch

data_collator = CustomDataCollator()

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=processor,
    data_collator=data_collator
)

# Train
trainer.train()


# Print memory usage
print(f"Allocated memory: {torch.cuda.memory_allocated('cuda:0') / 1e9:.2f} GB")
print(f"Cached memory: {torch.cuda.memory_reserved('cuda:0') / 1e9:.2f} GB")
print(f"Peak allocated memory: {torch.cuda.max_memory_allocated('cuda:0') / 1e9:.2f} GB")