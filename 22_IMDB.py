# 导入必要的库和模块
from datasets import load_dataset  # 从 Hugging Face Datasets 库加载数据集
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 导入预训练的分词器和模型
from transformers import TrainingArguments, Trainer  # 导入训练参数和 Trainer 类
from sklearn.metrics import accuracy_score  # 导入用于计算准确率的函数
import torch  # 导入 PyTorch 库，用于深度学习
import numpy as np  # 导入 NumPy 库，用于数值计算

# 设置设备为 GPU（如果可用）或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")  # 打印当前使用的设备

# 加载 IMDb 数据集
ds = load_dataset("stanfordnlp/imdb")  # 加载 Stanford NLP 提供的 IMDb 数据集
ds = ds.shuffle(seed=42)  # 打乱数据集顺序，设置种子以确保可重复性

# 加载预训练的分词器
model_name = "lvwerra/distilbert-imdb"  # 指定要使用的预训练模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从预训练模型中加载分词器

# 定义数据预处理函数
def preprocess_function(examples):
    """
    使用分词器对输入文本进行编码，包括截断和填充到最大长度。
    
    参数:
    examples (dict): 包含文本数据的字典，键为 "text"
    
    返回:
    dict: 包含编码后的输入 IDs 和注意力掩码
    """
    return tokenizer(
        examples["text"],  # 输入的文本
        truncation=True,  # 如果文本超过最大长度，则截断
        padding="max_length",  # 填充到最大长度
        max_length=512  # 设置最大长度为 512
    )

# 对数据集进行预处理
tokenized_ds = ds.map(preprocess_function, batched=True)  # 使用 map 函数批量应用预处理函数

# 加载预训练的序列分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,  # 使用与分词器相同的预训练模型
    num_labels=2  # 设置分类任务的标签数量为2（正面和负面）
).to(device)  # 将模型移动到指定设备（GPU 或 CPU）

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./imdb_classifier",  # 指定输出目录，用于保存模型和检查点
    learning_rate=5e-5,  # 设置学习率
    per_device_train_batch_size=32,  # 每个设备上的训练批次大小
    per_device_eval_batch_size=32,  # 每个设备上的评估批次大小
    num_train_epochs=2,  # 训练的总轮数
    weight_decay=0.01,  # 权重衰减，用于正则化
    evaluation_strategy="epoch",  # 评估策略，设置为每个epoch结束时评估
    save_strategy="epoch",  # 保存策略，设置为每个epoch结束时保存模型
    load_best_model_at_end=True,  # 在训练结束时加载表现最好的模型
    seed=42,  # 设置随机种子以确保可重复性
    lr_scheduler_type="linear",  # 设置学习率调度器类型为线性
)

# 定义优化器：AdamW，带有指定的 betas 和 epsilon
optimizer = torch.optim.AdamW(
    model.parameters(),  # 优化器将优化模型的所有参数
    lr=5e-5,  # 设置学习率，与 TrainingArguments 中一致
    betas=(0.9, 0.999),  # 设置 Adam 优化器的 betas 参数
    eps=1e-08,  # 设置 Adam 优化器的 epsilon 参数，增加数值稳定性
    weight_decay=0.01  # 设置权重衰减，防止过拟合
)

# 定义计算评估指标的函数
def compute_metrics(eval_pred):
    """
    计算预测结果的准确率。
    
    参数:
    eval_pred (tuple): 包含预测 logits 和真实标签的元组
    
    返回:
    dict: 包含准确率的字典
    """
    predictions, labels = eval_pred  # 解包预测结果和真实标签
    predictions = np.argmax(predictions, axis=1)  # 获取每个样本的预测类别（概率最大的索引）
    return {"accuracy": accuracy_score(labels, predictions)}  # 返回准确率

# 创建 Trainer 实例，用于训练和评估模型
trainer = Trainer(
    model=model,  # 需要训练的模型
    args=training_args,  # 训练参数
    train_dataset=tokenized_ds["train"],  # 训练数据集
    eval_dataset=tokenized_ds["test"],  # 评估数据集
    compute_metrics=compute_metrics,  # 评估指标函数
    optimizers=(optimizer, None)  # 指定优化器，第二个参数为学习率调度器，这里设为 None
)

# 开始训练模型
print("开始训练模型...")
trainer.train()  # 调用 Trainer 的 train 方法开始训练

# 评估模型性能
print("评估模型性能...")
eval_results = trainer.evaluate()  # 调用 Trainer 的 evaluate 方法进行评估
print(f"评估结果: {eval_results}")  # 打印评估结果

# 保存训练好的模型
print("保存模型...")
trainer.save_model("./imdb_classifier_final")  # 将模型保存到指定目录

# 预测部分（已注释掉）
# 以下部分的代码用于加载保存的模型并进行预测，但已被注释掉
# 如果需要使用，可以取消注释并运行

# # 加载tokenizer
# model_name = "lvwerra/distilbert-imdb"  # 使用与训练时相同的预训练模型
# tokenizer = AutoTokenizer.from_pretrained(model_name)  # 加载分词器

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 再次设置设备
# print(f"使用设备: {device}")  # 打印设备信息

# # 加载模型
# print("加载模型...")
# model = AutoModelForSequenceClassification.from_pretrained("./imdb_classifier_final").to(device)  # 从保存的路径加载模型并移动到设备

# # 预测部分
# print("开始预测...")
# text = [
#     "This movie was really good. I enjoyed it a lot.", 
#     "this Movie Makes Me Feel Very Uncomfortable.",
#     "The heroine who has lain down for 90 minutes is just as good as a stone statue. It is a convenient money taking project. The atmosphere is set off by the male protagonist who breaks through the fence layer by layer. The two supporting role, the old man and the black police, the mentally retarded man who was invented by the screenwriter, the high-quality physique was easily knocked down, the insight to deal with the crisis was too poor, and the two human flesh checkpoints with poor budget were broken in minutes. The kidnapping cases are too low, there is no battle of wits and courage, and there is no life hanging by a thread. The cost is small as a joke. The anesthetic acts as a seal, and the drug property recedes to do the countdown. The battle between the hero and the heroine is not as dynamic as the fairy tale drama. Fingers use five hair special effects to show their strength, which is a beautiful and absolutely perfunctory new director training project.",
#     "From 'Texas, Paris' to' Tokyo, Public Toilets', Wenders' theme has changed from Lonely to Alone. Lonely people are not necessarily shameful, they can engage in the same routine of washing toilets every day, and life does not lose its colors because of it. I may not be able to reach these days of low desires, low material possessions, and spiritual abundance, but my heart yearns for them. The world is big enough to accommodate countless small worlds. How cynical are those who believe that laborers do not deserve to enjoy a civilized life?"
# ]  # 定义要预测的文本列表
# encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")  # 对文本进行编码，返回 PyTorch tensors
# output = model(**encoded_text.to(device))  # 将编码后的文本输入模型，获取输出 logits
# print(output.logits)  # 打印模型的原始输出 logits
# print(torch.softmax(output.logits, dim=1))  # 打印经过 softmax 归一化后的概率分布