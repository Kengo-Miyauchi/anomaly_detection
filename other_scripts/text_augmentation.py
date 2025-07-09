from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# モデル・トークナイザーの読み込み
model_name = "rinna/japanese-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# テキスト生成関数
def generate_text(prompt, max_length=50, temperature=1.0, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 入力ファイルの読み込み（1行1文のテキストファイルを想定）
with open("reports.csv", "r", encoding="utf-8") as f:
    df = pd.read_csv(f)
    texts = df["text"].tolist()

# 各文に対して拡張文を生成
all_augmented = []

#import pdb; pdb.set_trace()
for text in texts:
    prompt = f"風車データに関する記述：「{text}」の意味を変えず言い換えてください。\n→ "
    augmented_sentences = []

    for _ in range(5):
        gen = generate_text(prompt, max_length=100)
        augmented = gen.replace(prompt, "").strip()
        augmented_sentences.append(augmented)

    print(f"\n元の文：{text}")
    print("拡張文：")
    for aug in augmented_sentences:
        print(" -", aug)

    all_augmented.append({
        "original": text,
        "augmented": augmented_sentences
    })

# 必要に応じてCSV保存（フラット形式）
df_out = pd.DataFrame([
    {"original": item["original"], "augmented": aug}
    for item in all_augmented
    for aug in item["augmented"]
])
df_out.to_csv("augmented_reports.csv", index=False, encoding="utf-8")
