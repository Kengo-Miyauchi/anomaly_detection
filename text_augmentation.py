from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# モデル・トークナイザーの読み込み
model_name = "rinna/japanese-gpt2-medium"  # 例：日本語GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

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

original_sentence = "この製品はとても良いです。"
prompt = f"文：「{original_sentence}」の意味を変えずに言い換えてください。\n→ "

generated = generate_text(prompt, max_length=60)
print("拡張文:", generated.replace(prompt, ""))

augmented_sentences = []
for _ in range(5):
    gen = generate_text(prompt, max_length=60)
    augmented_sentences.append(gen.replace(prompt, ""))

print("\n".join(augmented_sentences))
