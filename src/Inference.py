from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch


BASE_MODEL = "sarvamai/sarvam-1"
ADAPTER = "baze-il/sahaya-dpo"  


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()
print("✅ Sahaya ready!")


# inference function
def ask_sahaya(question, max_new_tokens=300):
    prompt = f"""You are Sahaya, a helpful assistant for Kerala government services.
Answer clearly in the same language the user asks.

User: {question}
Assistant:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()



if __name__ == "__main__":
    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = ask_sahaya(question)
        print(f"Sahaya: {answer}")