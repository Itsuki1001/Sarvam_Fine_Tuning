import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)



BASE_MODEL = "sarvamai/sarvam-1"
SFT_DATASET = "baze-il/sahaya-kerala-govt-sft"
DPO_DATASET = "baze-il/sahaya-kerala-govt-dpo"
SFT_OUTPUT = "baze-il/sahaya-sft"
DPO_OUTPUT = "baze-il/sahaya-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_dataset = load_dataset(SFT_DATASET, split="train")

def format_prompt(example):
    response = example["response"]
    response = response.replace("Step ", "\nStep ")
    response = response.replace("Documents:", "\nDocuments:")
    response = response.replace("Fee:", "\nFee:")
    response = response.replace("Time:", "\nTime:")
    text = f"""You are Sahaya, a helpful assistant for Kerala government services.
Answer clearly in the same language the user asks.

User: {example["prompt"]}
Assistant: {response}"""
    return {"text": text}

sft_dataset = sft_dataset.map(format_prompt)
sft_dataset = sft_dataset.remove_columns([col for col in sft_dataset.column_names if col != "text"])

sft_config = SFTConfig(
    output_dir="/tmp/sahaya_sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    dataset_text_field="text",
    max_length=512,
)

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    args=sft_config,
    processing_class=tokenizer,
)

sft_trainer.train()

model.push_to_hub(SFT_OUTPUT)
tokenizer.push_to_hub(SFT_OUTPUT)

del model
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    SFT_OUTPUT,
    is_trainable=True,
)

dpo_dataset = load_dataset(DPO_DATASET, split="train")

dpo_config = DPOConfig(
    output_dir="/tmp/sahaya_dpo",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    beta=0.1,
    max_length=512,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)

dpo_trainer.train()

model.push_to_hub(DPO_OUTPUT)
tokenizer.push_to_hub(DPO_OUTPUT)