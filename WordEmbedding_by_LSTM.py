import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import math
import csv

en_file = 'en.txt'
zh_file = 'zh.txt'

# 打开中文txt时增加encoding=utf8
with open(en_file,'r') as f:
    data = f.read()
# 处理中文txt时增加下面一句
# data = data.replace('。','').replace('、','').replace('“','').replace('“','').replace('’','').replace('‘','').replace('《','').replace('》','')
data = data.split('\n')
# 使用tokenizer分词器构建5000词的语料库
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
# num_words=5000已经小于训练集语料所有词了
tokenizer.fit_on_texts(data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data)
padded = pad_sequences(sequences, maxlen=25)
# print(padded[0])
# 生成训练数据集和标签
# 考虑设计nlp任务，将每一句的最后一个单词作为标签，除最后一词外的其他词作为训练数据。
# 以此来训练有意义的词向量
train_label = padded[:,-1].copy().astype(np.int)
train_data = np.delete(padded,np.s_[-1],axis=1)
# 将标签进行one-hot编码
train_label = ku.to_categorical(train_label,num_classes=5000)

# 构建基于LSTM的循环神经网络模型
# input_dim是词典大小
# output_dim是输出维度，即嵌入词向量的维度
# input_length是词的最大长度
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=5000, output_dim=50, input_length=24))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(5000,activation=tf.nn.softmax))
model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.categorical_crossentropy,metrics=['acc'])
model.summary()

history = model.fit(train_data,train_label,epochs=30,verbose=1)
model.save('LSTM_words_pool_30epochs.h5')
# 训练完整个模型，之后我们只需要用到模型的第一层即嵌入层，来观察词向量的生成情况
# 生成中间模型
model = tf.keras.models.load_model('LSTM_words_pool_30epochs.h5')
inter_model = keras.Model(
    inputs = model.input,
    outputs = model.get_layer("embedding").output
)

# 调用中间模型得到词向量，并将所有的词向量保存入csv文件
key = list(word_index.keys())
value = list(word_index.values())
embedding_value = []
for i in range(math.floor(5000/24)):
    tmp_value = value[24*i : 24*i+24]
    tmp_value = np.array(tmp_value)
    tmp = inter_model.predict(tmp_value.reshape(1,-1))[0]
    for j in range(24):
        embedding_value.append(list(tmp[j]))
tmp_value = np.array([i+4992 for i in range(8)]+[1 for i in range(16)])
tmp = inter_model.predict(tmp_value.reshape(1,-1))[0]
for i in range(8):
    embedding_value.append(list(tmp[i]))

key = np.array(key[:5000]).reshape(-1,1)
embedding_value = np.array(embedding_value)
data_total = np.concatenate((key,embedding_value),axis=1)
# 写入csv文件
with open('embedded_words_en.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for row in data_total:
        writer.writerow(row)


