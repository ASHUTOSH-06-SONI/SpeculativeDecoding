import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, login
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Dataset loader- this one ive asked gpt to 
# specifically give script for mac coz of some issues, mostly skill

def load_in22_conv_split(split="test", repo_id="ai4bharat/IN22-Conv"):
    token = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if token:
        try:
            login(token=token)
        except:
            pass

    api = HfApi()
    print(f"[INFO] Listing files in dataset repo: {repo_id}")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    parquet_files = [f for f in files if f.endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError("No parquet files found.")

    # Find all files matching split
    split_candidates = [
        f for f in parquet_files
        if split in f.lower() or f"/{split}-" in f or f"{split}-" in f
    ]

    # fallback
    if not split_candidates:
        split_candidates = parquet_files

    print("[INFO] Using parquet files:", split_candidates)

    downloaded = []
    for f in split_candidates:
        p = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset", use_auth_token=True)
        downloaded.append(p)

    dfs = []
    for p in downloaded:
        print("[INFO] Reading:", p)
        dfs.append(pd.read_parquet(p))

    df = pd.concat(dfs, ignore_index=True)
    print("[INFO] Final DataFrame size:", len(df))

    ds = Dataset.from_pandas(df)
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns("__index_level_0__")

    return ds


# 2. SIMPLE SPECULATIVE DECODING (or fallback sequential generation)

def generate(model, tokenizer, prompt, max_new_tokens=50):
    # GPT-2 max length = 1024
    MAX_INPUT = 900

    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_INPUT,
        return_tensors="pt"
    )

    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# 3. MAIN PIPELINE

def main():
    print(" LOADING DATASET  ")

    ds = load_in22_conv_split(split="test")  
    print(" LOADING MODEL + TOKENIZER ")
    model_name = "gpt2"           # small model coz my laptop can't lol 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(" RUNNING GENERATION ")
    for i in range(5):
        sample = ds[i]
        if "hin_Deva" in sample:
            prompt = sample["hin_Deva"]
        else:
            prompt = str(sample)

        print("\n--- SAMPLE", i+1, "INPUT --------------------")
        print(prompt)

        print("\n--- MODEL OUTPUT --------------------------")
        out = generate(model, tokenizer, prompt)
        print(out)
    print(" DONE ")


if __name__ == "__main__":
    main()

"""
for i in range(5):
    sample = ds[i]
    refer to ai4bharat ka dataset, in that,
    the language ka shortform is available, do this for any language
    or do it for all languages.
    if "hin_Deva" in sample:     
        prompt = sample["hin_Deva"]
    else:
        prompt = str(sample)

    print("\n--- SAMPLE", i+1, "INPUT --------------------")
    print(prompt)

    print("\n--- MODEL OUTPUT --------------------------")
    out = generate(model, tokenizer, prompt)
    print(out)

"""