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
    output = tokenizer(examples["text"], return_tensors="pt", truncation=True)
    max_length = max(len(ids) for ids in output["input_ids"])  # Find max length in this batch

    padded_input_ids = []

    for ids in output["input_ids"]:
        # Calculate the number of padding tokens needed
        num_padding_tokens = max_length - len(ids)
        # Pad the input_ids
        padded_ids = torch.cat([
            torch.full((num_padding_tokens,), 0, dtype=torch.long),  # Using 0 as the pad token ID
            torch.tensor(ids)
        ])

        padded_input_ids.append(padded_ids)

    # Convert list of padded input ids to a tensor and move to the specified device (e.g., CUDA)
    padded_input_ids = torch.stack(padded_input_ids).to("cuda:0")

    # Process the labels similarly, if necessary
    output["labels"] = torch.full_like(padded_input_ids, -100)
    for i, ids in enumerate(padded_input_ids):
        position = (ids == tokenizer.vocab["<s>"]).nonzero(as_tuple=True)[0][0]
        output["labels"][i, position:] = ids[position:]

    return {"input_ids": padded_input_ids, "labels": output["labels"]}



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
