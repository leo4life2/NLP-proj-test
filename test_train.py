import torch
from transformers import FuyuProcessor, FuyuForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig
from datasets import Dataset
import random
import string

# Function to generate random text
def generate_random_text(length=100):
    letters = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choice(letters) for _ in range(length))

# Parameters
num_examples = 100
max_length = 128
batch_size = 1

# Prepare dataset
texts = [generate_random_text(max_length) for _ in range(num_examples)]
data = {'text': texts}
dataset = Dataset.from_dict(data)

# Tokenize dataset
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
def tokenize_function(examples):
    return processor(examples["text"], truncation=True, max_length=max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model and use DataParallel for multi-GPU
model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", load_in_4bit=True)
model = torch.nn.DataParallel(model)
model.to('cuda')

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",          
    num_train_epochs=1,              
    per_device_train_batch_size=batch_size,  
    logging_dir='./logs',            
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

# Print memory usage (for each GPU)
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:")
    print(f"Allocated memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"Cached memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    print(f"Peak allocated memory: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats(i)
