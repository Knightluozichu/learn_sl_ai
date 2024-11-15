# %%导入 TensorFlow 和其他库
import tensorflow as tf

import numpy as np
import os
import time

# 下载莎士比亚数据集
# 更改以下行以在您自己的数据上运行此代码。

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 读取数据
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f'Length of text: {len(text)} characters')
print(text[:250])
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')
# %%
# 处理文本
# 将文本矢量化
# 在训练之前，您需要将字符串转换为数字表示形式。

# 该tf.keras.layers.StringLookup层可以将每个字符转换为数字 ID。它只需要先将文本拆分成标记即可。
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

# 现在创建tf.keras.layers.StringLookup图层：
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
# 它将标记转换为角色 ID
ids  = ids_from_chars(chars)
print(ids)
# %%
# 这些数字对应于vocab中的每个字符。将它们转换回字符：
# 由于本教程的目标是生成文本，因此反转此表示并从中恢复人类可读的字符串也很重要。为此，您可以使用tf.keras.layers.StringLookup(..., invert=True)。
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
# 该层从 ID 向量中恢复字符，并将它们作为tf.RaggedTensor字符返回：
chars = chars_from_ids(ids)
print(chars)
# %%
# 您可以tf.strings.reduce_join将字符重新连接成字符串。
tf.strings.reduce_join(chars, axis=-1).numpy()
print(chars)
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

'''
预测任务
给定一个字符或一个字符序列，最有可能的下一个字符是什么？
这是您训练模型要执行的任务。模型的输入将是字符序列，您训练模型来预测输出——每个时间步骤的下一个字符。

由于 RNN 维持依赖于先前看到的元素的内部状态，因此，给定到目前为止计算的所有字符，下一个字符是什么？
'''

# 创建训练示例和目标
# 接下来将文本分成示例序列。每个输入序列将包含seq_length来自文本的字符。

# 对于每个输入序列，相应的目标包含相同长度的文本，只是向右移动一个字符。

# 因此，将文本分成 块seq_length+1。例如，假设seq_length是 4，我们的文本是“Hello”。输入序列将是“Hell”，目标序列将是“ello”。

# 为此，首先使用tf.data.Dataset.from_tensor_slices函数将文本向量转换为字符索引流

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
print(all_ids)
# %%
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))
# %%
seq_length = 100

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))
# %%

'''
为了进行训练，您需要一(input, label)组对的数据集。其中input和 label是序列。
在每个时间步骤中，输入是当前字符，标签是下一个字符。

这是一个将序列作为输入、复制并移动它以对齐每个时间步的输入和标签的函数：
'''
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

split_input_target(list("Tensorflow"))
dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())
# %%
# 创建训练批次
# 您过去常常tf.data将文本拆分成可管理的序列。但在将这些数据输入模型之前，您需要对数据进行打乱并将其打包成批。
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

print(dataset)
# %%建立模型
# 本节将模型定义为keras.Model子类（有关详细信息，请参阅通过子类化创建新的图层和模型）。

# 这个模型有三个层：

# tf.keras.layers.Embedding：输入层。一个可训练的查找表，将每个字符 ID 映射到具有embedding_dim维度的向量；
# tf.keras.layers.GRU：一种具有大小的 RNN 类型units=rnn_units（您也可以在这里使用 LSTM 层。）
# tf.keras.layers.Dense：输出层，带有vocab_size输出。它为词汇表中的每个字符输出一个逻辑。这些是根据模型得出的每个字符的对数似然值。


# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    # 修复状态处理
    if states is None:
        states = tf.zeros([tf.shape(x)[0], self.gru.units])
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# 在上面的例子中，输入的序列长度是，100但模型可以在任意长度的输入上运行：
model.summary()
# %% 要从模型中获得实际预测，您需要从输出分布中抽样，以获取实际字符索引。此分布由字符词汇表上的 logits 定义。

# 尝试一下批次中的第一个示例：
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print(sampled_indices)

# 解码这些以查看该未经训练的模型预测的文本：
print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

# 训练模型
# 此时，该问题可以视为标准分类问题。给定前一个 RNN 状态和此时间步的输入，预测下一个字符的类别。

# 附加优化器和损失函数
# 在这种情况下，标准tf.keras.losses.sparse_categorical_crossentropy损失函数有效，因为它应用于预测的最后一个维度。

# 因为您的模型返回 logits，所以您需要设置from_logits标志。

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", example_batch_mean_loss)
# %%
# 新初始化的模型不应该对自己太有信心，输出的逻辑值应该都具有相似的量级。为了证实这一点，你可以检查平均损失的指数是否大约等于词汇量。损失高得多意味着模型对自己的错误答案很有把握，并且初始化得很糟糕：
tf.exp(example_batch_mean_loss).numpy()

# 使用该方法配置训练过程tf.keras.Model.compile。使用tf.keras.optimizers.Adam默认参数和损失函数
model.compile(optimizer='adam', loss=loss)

# 配置检查点
# 使用tf.keras.callbacks.ModelCheckpoint确保在训练期间保存检查点：
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 20

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# 生成文本
# 使用此模型生成文本的最简单方法是循环运行它，并在执行时跟踪模型的内部状态

# 每次调用模型时，您都会传入一些文本和内部状态。模型会返回下一个字符及其新状态的预测。将预测和状态传回去以继续生成文本。

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
# %%
'''
为了提高结果，最简单的方法就是进行更长时间的训练（尝试EPOCHS = 30）。

您还可以尝试不同的起始字符串，尝试添加另一个 RNN 层来提高模型的准确性，或者调整温度参数以生成或多或少随机的预测。

如果您希望模型更快地生成文本，最简单的方法是批量生成文本。在下面的示例中，模型生成 5 个输出所用的时间与上面生成 1 个输出所用的时间大致相同。
'''

start = time.time()
states = None
next_char = tf.constant(['ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result, '\n\n' + '_'*80)
print('\nRun time:', end - start)
# %%
# 导出生成器
# 该单步模型可以轻松保存和恢复，让您可以在任何接受的地方使用它tf.saved_model。

# tf.saved_model.save(one_step_model, 'one_step')
# TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
# %%
# one_step_reloaded = tf.saved_model.load('one_step')
# states = None
# next_char = tf.constant(['ROMEO:'])
# result = [next_char]

# for n in range(100):
#   next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
#   result.append(next_char)

# print(tf.strings.join(result)[0].numpy().decode("utf-8"))