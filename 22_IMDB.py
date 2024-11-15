from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import torch
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据集
ds = load_dataset("stanfordnlp/imdb")
ds = ds.shuffle(seed=42)

# 加载tokenizer
model_name = "lvwerra/distilbert-imdb"  # 使用基础模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# 对数据集进行预处理
tokenized_ds = ds.map(preprocess_function, batched=True)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./imdb_classifier",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,
    lr_scheduler_type="linear",
)

# optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

# 定义计算指标的函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

# 训练模型
print("开始训练模型...")
trainer.train()

# 评估模型
print("评估模型性能...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 保存模型
print("保存模型...")
trainer.save_model("./imdb_classifier_final")

# 预测-----------------------------------------------------------------------------------------------------------

# # 加载tokenizer
# model_name = "lvwerra/distilbert-imdb"  # 使用基础模型
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# # 加载模型
# print("加载模型...")
# model = AutoModelForSequenceClassification.from_pretrained("./imdb_classifier_final").to(device)

# # 预测 生成一些情感文本
# print("开始预测...")
# text = ["This movie was really good. I enjoyed it a lot.", 
#         "this Movie Makes Me Feel Very Uncomfortable.",
#         "The heroine who has lain down for 90 minutes is just as good as a stone statue. It is a convenient money taking project. The atmosphere is set off by the male protagonist who breaks through the fence layer by layer. The two supporting role, the old man and the black police, the mentally retarded man who was invented by the screenwriter, the high-quality physique was easily knocked down, the insight to deal with the crisis was too poor, and the two human flesh checkpoints with poor budget were broken in minutes. The kidnapping cases are too low, there is no battle of wits and courage, and there is no life hanging by a thread. The cost is small as a joke. The anesthetic acts as a seal, and the drug property recedes to do the countdown. The battle between the hero and the heroine is not as dynamic as the fairy tale drama. Fingers use five hair special effects to show their strength, which is a beautiful and absolutely perfunctory new director training project.",
#         "From 'Texas, Paris' to' Tokyo, Public Toilets', Wenders' theme has changed from Lonely to Alone. Lonely people are not necessarily shameful, they can engage in the same routine of washing toilets every day, and life does not lose its colors because of it. I may not be able to reach these days of low desires, low material possessions, and spiritual abundance, but my heart yearns for them. The world is big enough to accommodate countless small worlds. How cynical are those who believe that laborers do not deserve to enjoy a civilized life?"]
# encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
# output = model(**encoded_text.to(device))
# print(output.logits)
# print(torch.softmax(output.logits, dim=1))