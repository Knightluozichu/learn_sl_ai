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
model_name = "distilbert-base-uncased"  # 使用基础模型
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
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

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