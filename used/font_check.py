import matplotlib.font_manager as fm

# 利用可能なフォント一覧を出力（日本語対応を探す）
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    font_name = fm.FontProperties(fname=font).get_name()
    if any(kw in font_name for kw in ["Gothic", "Noto", "IPA", "Yu", "ヒラギノ", "ＭＳ"]):
        print(font_name)