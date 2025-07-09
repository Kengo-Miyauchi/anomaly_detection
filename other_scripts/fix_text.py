import pandas as pd
import jaconv
import re

# CSV読み込み
df = pd.read_csv("attributes.csv", encoding="utf-8")

# 列名整形関数（英数字はそのまま）
def clean_column(col):
    col = str(col).strip()  # 前後の空白
    col = re.sub(r'[ 　]+', '', col)  # 半角・全角スペース削除
    col = jaconv.h2z(col, kana=True, digit=False, ascii=False)  # 半角カタカナ → 全角
    return col

# 適用
new_columns = [clean_column(col) for col in df.columns]
df.columns = new_columns

# 保存（確認のため別名推奨）
df.to_csv("attributes_cleaned.csv", index=False, encoding="utf-8")

# 確認
for old, new in zip(df.columns, new_columns):
    if old != new:
        print(f"変換前: {old} → 変換後: {new}")
