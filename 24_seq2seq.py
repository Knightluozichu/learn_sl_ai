# %%
# Import generic wrappers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 


# Define the model repo
model_name = "Helsinki-NLP/opus-mt-zh-en" 


# Download pytorch model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 输入文本
input_text = "杰伊今天很开心，玩了一下午的游戏。这是什么游戏？英雄联盟，那就明天好好学习吧！"

# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt")

# 生成翻译
outputs = model.generate(**inputs)

# 解码并打印翻译结果
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
# %%
