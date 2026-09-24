# ==========================================
# ELIS AI - Concept Understanding Training
# ==========================================

!pip -q install transformers datasets accelerate peft

import json
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ------------------------------------------
# تنظیمات
# ------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_FILE = "elis_understanding.jsonl"

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------
# خواندن دیتاست
# ------------------------------------------

dataset = load_dataset(
    "json",
    data_files=DATASET_FILE,
    split="train"
)

print("تعداد نمونه‌ها:", len(dataset))


# ------------------------------------------
# تبدیل هر نمونه به متن آموزشی
# ------------------------------------------

def format_example(example):

    text = f"""### Instruction:
{example["instruction"]}

### User:
{example["input"]}

### Intent:
{example["intent"]}

### Context:
{example["context"]}

### ELIS:
{example["response"]}
"""

    return {"text": text}


dataset = dataset.map(format_example)


# ------------------------------------------
# Tokenizer
# ------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ------------------------------------------
# Tokenize
# ------------------------------------------

def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512
    )


tokenized = dataset.map(
    tokenize,
    batched=False,
    remove_columns=dataset.column_names
)


# ------------------------------------------
# مدل
# ------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
)

model.config.pad_token_id = tokenizer.pad_token_id


# ------------------------------------------
# Data Collator
# ------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ------------------------------------------
# تنظیمات آموزش
# ------------------------------------------

training_args = TrainingArguments(
    output_dir="./elis-understanding",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=5,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False
)


# ------------------------------------------
# Trainer
# ------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)


# ------------------------------------------
# شروع آموزش
# ------------------------------------------

print("\nشروع آموزش ELIS AI...\n")

trainer.train()


# ------------------------------------------
# ذخیره مدل
# ------------------------------------------

OUTPUT_DIR = "./elis-understanding-final"

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n================================")
print("آموزش تمام شد.")
print("مدل ذخیره شد در:")
print(OUTPUT_DIR)
print("================================")
