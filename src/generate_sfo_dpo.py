import json
import time
from datetime import datetime
from openai import OpenAI
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_KEY")
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)

client = OpenAI(api_key=api_key)


RAW_DATA_PATH = r"D:\Basil\Python\Sarvam_Fine_Tuning\Dataset\manual_raw_data.json"
SFT_DATASET = "baze-il/sahaya-kerala-govt-sft"
DPO_DATASET = "baze-il/sahaya-kerala-govt-dpo"


def call_openai(prompt, max_tokens=3000, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 10
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return None
    return None


def extract_json(text):
    try:
        text = text.strip()
        start = text.find('[')
        if start == -1:
            return None
        text = text[start:]
        try:
            return json.loads(text)
        except:
            pass
        last_complete = text.rfind('},')
        if last_complete == -1:
            last_complete = text.rfind('}')
        if last_complete != -1:
            fixed = text[:last_complete+1] + ']'
            try:
                return json.loads(fixed)
            except:
                pass
        return None
    except:
        return None


def generate_sft_pairs(service_name, english_content, malayalam_content):
    all_pairs = []

    batch_prompts = [
        {
            "focus": "English Q&A",
            "instruction": """Generate 10 Q&A pairs where:
- Questions are in English
- Answers are in English
- 5 procedure questions with numbered steps, 5 general queries (what is X, fees, documents, processing time)"""
        },
        {
            "focus": "Malayalam Q&A",
            "instruction": """Generate 10 Q&A pairs where:
- Questions are in Malayalam script
- Answers are in Malayalam script
- 5 procedure questions with numbered steps, 5 general queries"""
        },
        {
            "focus": "Manglish Q&A",
            "instruction": """Generate 10 Q&A pairs where:
- Questions are in Manglish (Malayalam written in English letters like WhatsApp style)
- Answers are in Manglish
- Example: "RC transfer cheyyaan enthu documents vendum?"
- 5 procedure questions with steps, 5 general queries"""
        },
        {
            "focus": "Cross-language Q&A",
            "instruction": """Generate 10 Q&A pairs:
- 3 pairs: English question asking to answer in Malayalam
- 3 pairs: Manglish question with English answer
- 2 pairs: Very casual Manglish like "evide pokkanam?" or "enthu vendum?"
- 2 pairs: General awareness questions in English"""
        }
    ]

    for i, batch in enumerate(batch_prompts):
        prompt = f"""You are creating training data for Sahaya — a Kerala government services assistant.

Service: {service_name}

English Reference:
---
{english_content[:3500]}
---

Malayalam Reference:
---
{malayalam_content[:3500]}
---

Task: {batch['focus']}
{batch['instruction']}

Rules:
- Procedure answers must use numbered steps:
  Step 1: ...
  Step 2: ...
  Step 3: ...
  Documents: X, Y, Z | Fee: ₹XX | Time: X days
- General query answers should be 2-4 sentences
- Include official portal links where relevant
- Use only information from the reference content

Return ONLY a valid JSON array:
[
  {{
    "prompt": "question here",
    "response": "answer here",
    "language": "english/malayalam/manglish/cross",
    "query_type": "procedure/general"
  }},
  ...
]"""

        print(f"  Batch {i+1}/4 [{batch['focus']}]...")
        result = call_openai(prompt)
        if not result:
            continue

        pairs = extract_json(result)
        if not pairs:
            print(f"  Parse failed batch {i+1}")
            continue

        for pair in pairs:
            pair['service'] = service_name
            pair['type'] = 'sft'
            pair['date'] = datetime.now().strftime("%Y-%m-%d")

        all_pairs.extend(pairs)
        time.sleep(2)

    print(f"  {len(all_pairs)} SFT pairs generated")
    return all_pairs


def generate_dpo_pairs(sft_pairs, service_name):
    all_dpo = []
    sample = sft_pairs[:20]

    for i in range(0, len(sample), 5):
        batch = sample[i:i+5]
        pairs_text = json.dumps(
            [{"prompt": p["prompt"], "chosen": p["response"]} for p in batch],
            ensure_ascii=False, indent=2
        )

        prompt = f"""For each Q&A pair, generate a BAD rejected response in the same language as the question.

Bad responses should be vague, missing steps, missing fees, missing documents, unhelpful.
- Malayalam question → bad Malayalam answer
- Manglish question → bad Manglish answer
- English question → bad English answer

Return ONLY a valid JSON array:
[
  {{
    "prompt": "...",
    "chosen": "...",
    "rejected": "bad answer in same language"
  }},
  ...
]

Input:
{pairs_text}"""

        result = call_openai(prompt, max_tokens=2000)
        if not result:
            continue

        pairs = extract_json(result)
        if not pairs:
            continue

        for pair in pairs:
            pair['service'] = service_name
            pair['type'] = 'dpo'
            pair['date'] = datetime.now().strftime("%Y-%m-%d")

        all_dpo.extend(pairs)
        time.sleep(2)

    print(f"  {len(all_dpo)} DPO pairs generated")
    return all_dpo


with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} services")

all_sft = []
all_dpo = []

for service_name, content in raw_data.items():
    print(f"\n{service_name.upper()}")

    sft_pairs = generate_sft_pairs(service_name, content['english'], content['malayalam'])
    all_sft.extend(sft_pairs)
    time.sleep(3)

    dpo_pairs = generate_dpo_pairs(sft_pairs, service_name)
    all_dpo.extend(dpo_pairs)
    time.sleep(3)

print(f"\nSFT: {len(all_sft)} pairs")
print(f"DPO: {len(all_dpo)} pairs")

sft_dataset = Dataset.from_list(all_sft)
sft_dataset.push_to_hub(SFT_DATASET, private=False)
print(f"SFT pushed to {SFT_DATASET}")

dpo_dataset = Dataset.from_list(all_dpo)
dpo_dataset.push_to_hub(DPO_DATASET, private=False)
print(f"DPO pushed to {DPO_DATASET}")