import torch
from transformers import FuyuProcessor, FuyuForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig

# Load Yelp Reviews dataset
dataset = load_dataset("yelp_review_full")

# Use FuyuProcessor for tokenization
# processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b")

def process_function(examples):
    output = tokenizer(examples["text"])
    # Create a mask where the condition is True
    mask = output["input_ids"] == tokenizer.vocab["<s>"]
    # Find the position of the first occurrence of <s>
    position = mask.nonzero(as_tuple=True)[0][0]

    output["labels"] = torch.full_like(output["input_ids"], -100)
    output["labels"][position:] = output["input_ids"][position:]


tokenized_datasets = dataset.map(process_function, batched=True)

# Create smaller subsets (optional)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

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
    tokenizer=tokenizer
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
