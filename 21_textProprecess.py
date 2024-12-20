# %%
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
# %%
'''
1 认识文本预处理
学习目标¶
了解文本预处理相关内容
1 文本预处理及其作用¶
文本语料在输送给模型前一般需要一系列的预处理工作, 才能符合模型输入的要求, 如: 将文本转化成模型需要的张量, 规范张量的尺寸等, 而且科学的文本预处理环节还将有效指导模型超参数的选择, 提升模型的评估指标.
2 文本预处理中包含的主要环节¶
文本处理的基本方法
文本张量表示方法
文本语料的数据分析
文本特征处理
数据增强方法
2.1 文本处理的基本方法¶
分词
词性标注
命名实体识别
2.2 文本张量表示方法¶
one-hot编码
Word2vec
Word Embedding
2.3 文本语料的数据分析¶
标签数量分布
句子长度分布
词频统计与关键词词云
2.4 文本特征处理¶
添加n-gram特征
文本长度规范
2.5 数据增强方法¶
回译数据增强法
2.6 重要说明¶
在实际生产应用中, 我们最常使用的两种语言是中文和英文，因此文本预处理部分的内容都将针对这两种语言进行讲解.
'''

# 导入用于对象保存与加载的joblib
import joblib
# 导入keras中的词汇映射器Tokenizer
# from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.text import Tokenizer
# 假定vocab为语料集所有不同词汇集合
vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
# 实例化一个词汇映射器对象
t = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0]*len(vocab)
    # 使用映射器转化现有文本数据, 每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)

# 使用joblib工具保存映射器, 以便之后使用
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)


# %% 获取数据文件夹
# 获取文件
import os, sys

# 获取当前文件的路径
def get_current_path():
    return os.path.dirname(os.path.realpath(__file__))

news_path = os.path.join(get_current_path(), 'data','ag_news')
# 检查文件夹是否存在
if not os.path.exists(news_path):
    print('文件夹不存在')
    sys.exit(0)
else:
    print('文件夹存在')
    print(news_path)



# %% 加载数据
from torchtext.datasets.text_classification import _csv_iterator, _create_data_from_iterator, TextClassificationDataset
from torchtext.utils import extract_archive
from torchtext.vocab import build_vocab_from_iterator, Vocab

def setup_datasets(ngrams=2, vocab_train=None, vocab_test=None, include_unk=False):
    train_csv_path = os.path.join(news_path, 'train.csv')
    test_csv_path = os.path.join(news_path, 'test.csv')

    if vocab_train is None:
        vocab_train = build_vocab_from_iterator(_csv_iterator(train_csv_path,ngrams))
    else:
        if not isinstance(vocab_train, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    if vocab_test is None:
        vocab_test = build_vocab_from_iterator(_csv_iterator(test_csv_path,ngrams))
    else:
        if not isinstance(vocab_test, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    
    train_data, train_labels = _create_data_from_iterator(vocab_train, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    test_data, test_labels = _create_data_from_iterator(vocab_test, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    
    return (TextClassificationDataset(vocab_train, train_data, train_labels), 
            TextClassificationDataset(vocab_test, test_data, test_labels))


train_dataset, test_dataset = setup_datasets(ngrams=2)
print('train_dataset:', train_dataset)

# %% 
# 第一步: 构建带有Embedding层的文本分类模型.
# 第二步: 对数据进行batch处理.
# 第三步: 构建训练与验证函数.
# 第四步: 进行模型训练和验证.
# 第五步: 查看embedding层嵌入的词向量.

# %% 第一步: 构建带有Embedding层的文本分类模型
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE= 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """初始化权重函数"""
        # 指定初始权重的取值范围数
        initrange = 0.5
        # 各层的权重参数都是初始化为均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def forward(self, text):
        """
        :param text: 文本数值映射后的结果
        :return: 与类别数尺寸相同的张量, 用以判断文本类别
        """
        # 获得embedding的结果embedded
        # >>> embedded.shape
        # (m, 32) 其中m是BATCH_SIZE大小的数据中词汇总数
        embedded = self.embedding(text)
        # 接下来我们需要将(m, 32)转化成(BATCH_SIZE, 32)
        # 以便通过fc层后能计算相应的损失
        # 首先, 我们已知m的值远大于BATCH_SIZE=16,
        # 用m整除BATCH_SIZE, 获得m中共包含c个BATCH_SIZE
        c = embedded.size(0) // BATCH_SIZE
        # 之后再从embedded中取c*BATCH_SIZE个向量得到新的embedded
        # 这个新的embedded中的向量个数可以整除BATCH_SIZE
        embedded = embedded[:BATCH_SIZE*c]
        # 因为我们想利用平均池化的方法求embedded中指定行数的列的平均数,
        # 但平均池化方法是作用在行上的, 并且需要3维输入
        # 因此我们对新的embedded进行转置并拓展维度
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        # 然后就是调用平均池化的方法, 并且核的大小为c
        # 即取每c的元素计算一次均值作为结果
        embedded = F.avg_pool1d(embedded, kernel_size=c)
        # 最后，还需要减去新增的维度, 然后转置回去输送给fc层
        return self.fc(embedded[0].transpose(1, 0))
    

# 获得整个语料包含的不同词汇总数
VOCAB_SIZE = len(train_dataset.get_vocab())
# 指定词嵌入维度
EMBED_DIM = 32
# 获得类别总数
NUN_CLASS = len(train_dataset.get_labels())
# 实例化模型
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


def generate_batch(batch):
    """
    description: 生成batch数据函数
    :param batch: 由样本张量和对应标签的元组组成的batch_size大小的列表
                  形如:
                  [(label1, sample1), (lable2, sample2), ..., (labelN, sampleN)]
    return: 样本张量和标签各自的列表形式(张量)
             形如:
             text = tensor([sample1, sample2, ..., sampleN])
             label = tensor([label1, label2, ..., labelN])
    """
    # 从batch中获得标签张量
    label = torch.tensor([entry[0] for entry in batch])
    # 从batch中获得样本张量
    text = [entry[1] for entry in batch]
    text = torch.cat(text)
    # 返回结果
    return text, label

# 假设一个输入:
batch = [(1, torch.tensor([3, 23, 2, 8])), (0, torch.tensor([3, 45, 21, 6]))]
res = generate_batch(batch)
print(res)


# 导入torch中的数据加载器方法
from torch.utils.data import DataLoader

def train(train_data):
    """模型训练函数"""
    # 初始化训练损失和准确率为0
    train_loss = 0
    train_acc = 0

    # 使用数据加载器生成BATCH_SIZE大小的数据进行批次训练
    # data就是N多个generate_batch函数处理后的BATCH_SIZE大小的数据生成器
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)

    # 对data进行循环遍历, 使用每个batch的数据进行参数更新
    for i, (text, cls) in enumerate(data):
        # 设置优化器初始梯度为0
        optimizer.zero_grad()
        # 模型输入一个批次数据, 获得输出
        output = model(text)
        # 根据真实标签与模型输出计算损失
        loss = criterion(output, cls)
        # 将该批次的损失加到总损失中
        train_loss += loss.item()
        # 误差反向传播
        loss.backward()
        # 参数进行更新
        optimizer.step()
        # 将该批次的准确率加到总准确率中
        train_acc += (output.argmax(1) == cls).sum().item()

    # 调整优化器学习率  
    scheduler.step()

    # 返回本轮训练的平均损失和平均准确率
    return train_loss / len(train_data), train_acc / len(train_data)

def valid(valid_data):
    """模型验证函数"""
    # 初始化验证损失和准确率为0
    loss = 0
    acc = 0

    # 和训练相同, 使用DataLoader获得训练数据生成器
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    # 按批次取出数据验证
    for text, cls in data:
        # 验证阶段, 不再求解梯度
        with torch.no_grad():
            # 使用模型获得输出
            output = model(text)
            # 计算损失
            loss = criterion(output, cls)
            # 将损失和准确率加到总损失和准确率中
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    # 返回本轮验证的平均损失和平均准确率
    return loss / len(valid_data), acc / len(valid_data)


# 导入时间工具包
import time

# 导入数据随机划分方法工具
from torch.utils.data.dataset import random_split

# 指定训练轮数
N_EPOCHS = 10

# 定义初始的验证损失
min_valid_loss = float('inf')

# 选择损失函数, 这里选择预定义的交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss().to(device)
# 选择随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
# 选择优化器步长调节方法StepLR, 用来衰减学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# 从train_dataset取出0.95作为训练集, 先取其长度
train_len = int(len(train_dataset) * 0.95)

# 然后使用random_split进行乱序划分, 得到对应的训练集和验证集
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# 开始每一轮训练
for epoch in range(N_EPOCHS):
    # 记录概论训练的开始时间
    start_time = time.time()
    # 调用train和valid函数得到训练和验证的平均损失, 平均准确率
    train_loss, train_acc = train(sub_train_)
    valid_loss, valid_acc = valid(sub_valid_)

    # 计算训练和验证的总耗时(秒)
    secs = int(time.time() - start_time)
    # 用分钟和秒表示
    mins = secs / 60
    secs = secs % 60

    # 打印训练和验证耗时，平均损失，平均准确率
    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

# %%
# 打印从模型的状态字典中获得的Embedding矩阵
print(model.state_dict()['embedding.weight'])

# %%
