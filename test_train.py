import torch
from transformers import FuyuProcessor, FuyuForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig

# Load Yelp Reviews dataset
dataset = load_dataset("yelp_review_full")
max_length = 128  # Define max_length according to your needs

# Use FuyuProcessor for tokenization
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")

def tokenize_function(examples):
    return processor(text=examples["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create smaller subsets (optional)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Load model
model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", load_in_4bit=True)

# LoRA configuration
lora_config = LoraConfig(
    target_modules=["query_key_value"],
    init_lora_weights=False
)
model.add_adapter(lora_config, adapter_name="lora")

# Use DataParallel for multi-GPU
model = torch.nn.DataParallel(model)
model.to('cuda')

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",          
    num_train_epochs=1,              
    per_device_train_batch_size=1,  
    logging_dir='./logs',            
    logging_steps=10,
    no_cuda=False  # Ensure CUDA is enabled
)

# Initialize Trainer with default data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=processor
)

# Train
trainer.train()

# Print memory usage for each GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:")
    print(f"Allocated memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"Cached memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    print(f"Peak allocated memory: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats(i)
