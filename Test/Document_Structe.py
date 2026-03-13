import os
import json

DOCS_PATH = r"D:\Basil\Python\Sarvam_Fine_Tuning\Documents"
SAVE_PATH = r"D:\Basil\Python\Sarvam_Fine_Tuning\Test"


NAME_MAP = {
    "adhar adress update":  "aadhaar",
    "adhar_adr":            "aadhaar",      
    "adhar_adr_malayalam":  "aadhaar",
    "adhar_adrayalam":      "aadhaar",
    "apply_passprt":        "passport",
    "file_itr":             "income_tax",
    "pan_apply":            "pan_card",
    "puc":                  "puc",
    "rc_transfer":          "rc_transfer",
    "rc_trf":               "rc_transfer",
    "re_reg":               "re_registration",
    "voters_id":            "voter_id",
}

raw_data = {}

for filename in os.listdir(DOCS_PATH):
    if not filename.endswith(".txt"):
        continue
    
    filepath = os.path.join(DOCS_PATH, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    name = filename.replace(".txt", "").strip()
    is_malayalam = name.lower().endswith("_mal") or name.lower().endswith("_malayalam") or name.lower().endswith("ayalam")
    
    # normalize name
    clean_name = name.lower()
    clean_name = clean_name.replace("_mal", "").replace("_malayalam", "").replace("ayalam", "").strip()
    
    # map to standard service name
    service_name = NAME_MAP.get(clean_name, clean_name)
    lang = "malayalam" if is_malayalam else "english"
    
    if service_name not in raw_data:
        raw_data[service_name] = {"english": "", "malayalam": ""}
    
    raw_data[service_name][lang] = content
    print(f" {filename} → [{lang}] {service_name}")

print(f"\n Total services: {len(raw_data)}")
for service, content in raw_data.items():
    en = len(content['english'])
    ml = len(content['malayalam'])
    flag = "✅" if en > 0 and ml > 0 else "⚠️"
    print(f"  {flag} {service}: EN={en} chars | ML={ml} chars")

# save
with open(f"{SAVE_PATH}/manual_raw_data.json", "w", encoding="utf-8") as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=2)

print(f"\n Saved to manual_raw_data.json")