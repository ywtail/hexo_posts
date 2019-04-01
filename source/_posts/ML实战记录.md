---
title: ML实战记录
date: 2019-03-01 21:16:22
tags: [MachineLearning,python]
categories: MachineLearning
---

详细过程见[达观杯文本智能处理](https://github.com/ywtail/ML/blob/master/Datawhale/%E8%BE%BE%E8%A7%82%E6%9D%AF%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86.ipynb)



提交记录

![达观杯文本智能处理提交记录](/images/达观杯文本智能处理提交记录.jpg)



使用LSTM（Long short-term memory, LSTM），lstm参数(https://keras.io/zh/layers/recurrent/)

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

**参数**

- **units**: 正整数，输出空间的维度。
- **activation**: 要使用的激活函数 (详见 [activations](https://keras.io/zh/activations/))。 如果传入 `None`，则不使用激活函数 (即 线性激活：`a(x) = x`)。
- **recurrent_activation**: 用于循环时间步的激活函数 (详见 [activations](https://keras.io/zh/activations/))。 默认：分段线性近似 sigmoid (`hard_sigmoid`)。 如果传入 `None`，则不使用激活函数 (即 线性激活：`a(x) = x`)。
- **use_bias**: 布尔值，该层是否使用偏置向量。
- **kernel_initializer**: `kernel` 权值矩阵的初始化器， 用于输入的线性转换 (详见 [initializers](https://keras.io/zh/initializers/))。
- **recurrent_initializer**: `recurrent_kernel` 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 [initializers](https://keras.io/zh/initializers/))。
- **bias_initializer**:偏置向量的初始化器 (详见[initializers](https://keras.io/zh/initializers/)).
- **unit_forget_bias**: 布尔值。 如果为 True，初始化时，将忘记门的偏置加 1。 将其设置为 True 同时还会强制 `bias_initializer="zeros"`。 这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
- **recurrent_regularizer**: 运用到 `recurrent_kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
- **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
- **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
- **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
- **recurrent_constraint**: 运用到 `recurrent_kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
- **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
- **dropout**: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
- **recurrent_dropout**: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
- **implementation**: 实现模式，1 或 2。 模式 1 将把它的操作结构化为更多的小的点积和加法操作， 而模式 2 将把它们分批到更少，更大的操作中。 这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- **return_sequences**: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- **return_state**: 布尔值。除了输出之外是否返回最后一个状态。
- **go_backwards**: 布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
- **stateful**: 布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态。
- **unroll**: 布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。 展开只适用于短序列。

代码参考：https://github.com/Heitao5200/DGB/blob/master/model/model_code/RCNN.py

数据处理，输入为中心词、左词和右词

```python
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=doc_len)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=doc_len)

left_train_word_ids = [[len(vocab)] + x[:-1] for x in X_train_word_ids]
left_test_word_ids = [[len(vocab)] + x[:-1] for x in X_test_word_ids]
right_train_word_ids = [x[1:] + [len(vocab)] for x in X_train_word_ids]
right_test_word_ids = [x[1:] + [len(vocab)] for x in X_test_word_ids]

left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=doc_len)
left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=doc_len)
right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=doc_len)
right_test_padded_seqs = pad_sequences(right_test_word_ids, maxlen=doc_len)

document = Input(shape = (doc_len, ), dtype = "int32")
left_context = Input(shape = (doc_len, ), dtype = "int32")
right_context = Input(shape = (doc_len, ), dtype = "int32")

# embedding
embedder = Embedding(len(vocab) + 1, embedding_dim, input_length = doc_len)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)
```

构建模型

```python
forward = LSTM(256, return_sequences = True)(l_embedding)
backward = LSTM(256, return_sequences = True, go_backwards = True)(r_embedding)
together = concatenate([forward, doc_embedding, backward], axis = 2)

semantic = TimeDistributed(Dense(128, activation = "tanh"))(together)
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (128, ))(semantic)
output = Dense(19, activation = "softmax")(pool_rnn) 
model = Model(inputs = [document, left_context, right_context], outputs = output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit([X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs],y_train,
           batch_size=32,
           epochs=1,
           validation_data=([X_test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs], y_test))
```

评价及预测

```python
score = model.evaluate(X_test_padded_seqs, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## 特征转换
xx_test_word_ids = tokenizer.texts_to_sequences(df_test['word_seg'])
xx_test_padded_seqs = pad_sequences(xx_test_word_ids, maxlen=doc_len)

## 预测
pred_prob = model.predict(xx_test_padded_seqs)
pred = pred_prob.argmax(axis=1)
```



模型融合

```python
models = [KNeighborsClassifier(n_neighbors=5,n_jobs = -1),RandomForestClassifier(random_state=0, n_jobs=-1,n_estimators = 100, max_depth = 3),XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,n_estimators = 100, max_depth = 3)]

S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, mode='oof_pred_bag', needs_proba=False, save_dir=None, metric=accuracy_score, n_folds=4, stratified=True, shuffle=True, random_state=0, verbose=2)
```

各参数含义如下

>The stacking function takes several inputs:

- **models**: the first level models we defined earlier
- **X_train, y_train, X_test**: our data
- **regression**: Boolean indicating whether we want to use the function for regression. In our case set to False since this is a classification
- **mode:** using the earlier describe out-of-fold during cross-validation
- **needs_proba**: Boolean indicating whether you need the probabilities of class labels
- **save_dir**: save the result to directory Boolean
- **metric**: what evaluation metric to use (we imported the accuracy_score in the beginning)
- **n_folds**: how many folds to use for cross-validation
- **stratified**: whether to use stratified cross-validation
- **shuffle**: whether to shuffle the data
- **random_state**: setting a random state for reproducibility
- **verbose**: 2 here refers to printing all info

最终使用XGB预测

```python
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3) 
model = model.fit(S_train, y_train) 
y_pred = model.predict(S_test)
```

