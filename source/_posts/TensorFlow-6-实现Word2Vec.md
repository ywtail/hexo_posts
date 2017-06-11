---
title: 'TensorFlow (6): 实现Word2Vec'
date: 2017-06-09 09:34:25
tags: [TensorFlow,MachineLearning,python]
categories: TensorFlow
top: 2
---

### Word2Vec 简介

Word2Vec 也称 Word Embeddings，中文比较普遍的叫法是“词向量”或“词嵌入”。Word2Vec 是一个可以将中文词转为向量形式表达（Vector Representations）的模型。

#### 为什么要把字词转为向量

图像、音频等数据天然可以编码并存储为稠密向量的形式，比如图片是像素点的稠密矩阵，音频可以转为声音信号的频谱数据。自然语言在 Word2Vec 出现之前，通常将字词转成离散的单独的符号，例如 "cat" 一词或可表示为 `Id537` ，而 "dog" 一词或可表示为 `Id143`。

这些符号编码毫无规律，无法提供不同词汇之间可能存在的关联信息。换句话说，在处理关于 "dogs" 一词的信息时，模型将无法利用已知的关于 "cats" 的信息（例如，它们都是动物，有四条腿，可作为宠物等等）。可见，将词汇表达为上述的独立离散符号将进一步导致数据稀疏，使我们在训练统计模型时不得不寻求更多的数据。而词汇的向量表示将克服上述的难题。

[向量空间模型](https://en.wikipedia.org/wiki/Vector_space_model) (Vector Space Models，VSMs)将词汇表达（嵌套）于一个连续的向量空间中，语义近似的词汇被映射为相邻的数据点。向量空间模型在自然语言处理领域中有着漫长且丰富的历史，不过几乎所有利用这一模型的方法都依赖于 [分布式假设](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)，其核心思想为**出现于上下文情景中的词汇都有相类似的语义**。采用这一假设的研究方法大致分为以下两类：*基于计数的方法* (e.g. [潜在语义分析](https://en.wikipedia.org/wiki/Latent_semantic_analysis))， 和 *预测方法* (e.g. [神经概率化语言模型](http://www.scholarpedia.org/article/Neural_net_language_models)).

其中它们的区别在如下论文中又详细阐述 [Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf)，不过简而言之：基于计数的方法计算某词汇与其邻近词汇在一个大型语料库中共同出现的频率及其他统计量，然后将这些统计量映射到一个小型且稠密的向量中。预测方法则试图直接从某词汇的邻近词汇对其进行预测，在此过程中利用已经学习到的小型且稠密的*嵌套向量*。

Word2vec 是一种可以进行高效率词嵌套学习的预测模型。其两种变体分别为：连续词袋模型（Continuous Bag of Words，CBOW）及 Skip-Gram 模型。从算法角度看，这两种方法非常相似，其区别为 CBOW 根据源词上下文词汇（'the cat sits on the'）来预测目标词汇（例如，‘mat’），而 Skip-Gram 模型做法相反，它通过目标词汇来预测源词汇。CBOW 对小型数据比较合适，而 Skip-Gram 在大型语料中表现得更好。

预测模型通常使用最大似然的方法，在给定前面的语句 h 的情况下，最大化目标词汇 w 的概率。但它存在的一个比较严重的问题是计算量非常大，需要计算词汇表中所有单词出现的可能性。在 Word2Vec 的 CBOW 模型中，不需要计算完整的概率模型，只需要训练一个二元的分类模型，用来区分真实的目标词汇和编造的词汇（噪声）这两类。用这种少量噪声词汇来估计的方法，类似于蒙特卡洛模拟。

当模型预测真实的目标词汇为高概率，同时预测其他噪声词汇为低概率时，我们训练的学习目标就被最优化了。用编造的噪声词汇训练的方法被称为 负采样（ `Negative Sampling`），用这种方法计算 loss function 的效率非常高，我们只需要计算随机选择的 k 个词汇而非词汇表中的全部词汇，因此训练速度非常快。在实际中，我们使用 `Noise-Contrastive Estimation(NCE) Loss `，同时在 TensorFlow 中也有 `tf.nn.nce_loss()` 直接实现了这个 loss。

在本节中我们将主要使用 Skip-Gram 模式的 Word2Vec。

更具体的信息参见：[TensorFow 中国社区：Vector Representations of Words](http://www.tensorfly.cn/tfdoc/tutorials/word2vec.html)

#### Skip-gram 模型

下面来看一下这个数据集

`the quick brown fox jumped over the lazy dog`

我们首先对一些单词以及它们的上下文环境建立一个数据集。我们可以以任何合理的方式定义‘上下文’，而通常上这个方式是根据文字的句法语境的（使用语法原理的方式处理当前目标单词可以看一下这篇文献 [Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)，比如说把目标单词左边的内容当做一个‘上下文’，或者以目标单词右边的内容，等等。现在我们把目标单词的左右单词视作一个上下文， 使用大小为1的窗口，这样就得到这样一个由`(上下文, 目标单词)` 组成的数据集：

`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`

前文提到Skip-Gram模型是把目标单词和上下文颠倒过来，所以在这个问题中，举个例子，就是用'quick'来预测 'the' 和 'brown' ，用 'brown' 预测 'quick' 和 'brown' 。因此这个数据集就变成由`(输入, 输出)`组成的：

`(quick, the), (quick, brown), (brown, quick), (brown, fox), ...`

目标函数通常是对整个数据集建立的，但是本问题中要对每一个样本（或者是一个`batch_size` 很小的样本集，通常设置为`16 <= batch_size <= 512`）在同一时间执行特别的操作，称之为[随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD)。

### 构造训练样本

实现 Word2Vec 首先需要构造训练样本。以 `the quick brown fox jumped over the lazy dog` 为例，我们需要构造一个语境与目标词汇的映射关系，假设我们的滑动窗口尺寸为 1，则语境包括一个单词左边和右边的词汇，可以制造的映射关系包括 `[the, brown] -> quick, [quick, fox] -> brown, [brown, jumped] -> fox` 等。

因为 Skip-Gram 模型是从目标词汇预测语境，所以训练样本不再是 [the, brown] -> quick，而是 quick -> the 和 quick -> brown。我们的数据集就变为了 (quick, the)、(quick, brown)、(brown, quick)、(brown, fox) 等。

我们训练时，希望模型能从目标词汇 quick 预测出语境 the，同时也需要制造随机的词汇作为负样本（噪声），我们希望预测的概率分布在正样本 the 上尽可能大，而在随机产生的负样本上尽可能小。这里的做法就是通过优化算法（例如 SGC）来更新模型中的 Word Embedding 的参数，让概率分布的损失函数（NCE Loss）尽可能小。这样每个单词的 Embedded Vector 就会随着就训练过程不断调整，直到出于一个最合适语料的空间位置。这样我们的损失函数最小，最符合语料，同时预测出正确单词的概率也最高。

#### 下载数据集

数据集获取有两种方法

- 在浏览器地址栏输入 [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip) 下载数据的压缩文件。

- 使用 urllib.urlretrieve 下载数据的压缩文件，并核对尺寸，如果已经下载了文件则跳过。下载成功提示 `('Found and verified', 'text8.zip')`

  ```python
  def maybe_download(filename, expected_bytes):
      if not os.path.exists(filename):
          filename, _ = urllib.urlretrieve(url + filename, filename)
      statinfo = os.stat(filename)
      if statinfo.st_size == expected_bytes:
          print('Found and verified', filename)
      else:
          print(statinfo.st_size)
          raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
      return filename

  filename = maybe_download('text8.zip', 31344016)
  ```

#### 解压并转为列表

接下来解压（使用 zipfile.ZipFile 函数）下载的压缩文件，并使用 tf.compat.as_str 将数据转成单词的列表。

```python
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print 'Data size', len(words)
```

通过输出知道数据最后被转为了一个包含 17005207 个单词的列表。

#### 创建词汇表

使用 collections.Counter 统计单词的频数，然后使用 most_common 方法获取词频数最高的 50000 个单词加入词汇表。

因为 python 中字典查询复杂度为 O(1)，性能非常好，所以再创建字典 dictionary，将词频最高的50000 个词汇放入 dictionary 中，以便快速查询。

接下来将全部单词转为编号（以频数排序的编号），top 50000 之外的词，认定其为 Unkown（未知），将其编号为0。

返回

- data：转换后的编码
- count：每个单词的频数统计
- dictionary：词汇表（词：编码）
- reverse_dictionary：词汇表的反转形式（编码：词）

```python
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]  # 前面是词汇，后面是出现的次数
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()  # 转换后的编码：如果出现在 dictionary 中，数量作为编号，不出现 0 作为编号
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
```

#### 生成 Word2Vec 训练样本

根据前面提到的 Skip-Gram 模式（从目标单词反推语境），将原始数据 `the quick brown fox jumped over the lazy dog` 转为(quick, the)、(quick, brown)、(brown, quick)、(brown, fox) 等样本。

定义 generate_batch 函数来生成训练用的 batch 数据。 其中 batch_size 为 batch 的大小；skip_window 为单词间最远可以联系到的距离，设为 1 代表只能跟紧相邻的一个单词生成样本，例如 quick 只能生成 (quick, the) 和 (quick, brown)；num_skips 为对每个目标单词提取的样本数，它不能大于 skip_window 的两倍，并且 batch_size 必须是它的整数倍（为了确保每个 batch 包含了一个词汇对应的所有样本）。

我们定义单词序号 data_index 为 global 变量，因为会反复调用 generate_batch，所以要确保 data_index 可以在函数 generate_batch 中被修改。

我么也用 assert 确保 skip_window 和 num_skips 满足前面提到的条件

然后用 np.ndarray 将 batch 和 labels 初始化为数组。

这定义span为对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词，因此  `span=2*skip_window+1`

并创建最大容量为 span 的 deque（双向队列），在用 append 对 deque 添加变量时，只会保留最后插入的 span 个变量

接下来从 data_index 开始，把 span 个单词顺序读入 buffer 作为初始值，buffer 中存的是词的编号。因为 buffer 是容量为 span 的 deque，所以此时 buffer 已经充满，后续数据将替换掉前面的数据。

然后我们进入第一层循环（次数为batch_size // num_skips），每次循环内对一个目标单词生成样本。现在 buffer 中是目标单词和所有相关单词，我们定义target=skip_window，即 buffer 中第 skip_window 个单词为目标单词。然后定义生成样本时需要避免的单词列表 targets_to_avoid，这个列表开始包括第 skip_window 个单词（即目标单词），因为我们要预测的是语境单词，不包括目标单词本身。

接下来进入第二层循环（次数为 num_skips），每次循环对一个语境单词生成样本， 先产生一个随机数，直到随机数不在 targets_to_avoid 中，就可以将之作为语境单词。feature 是目标词汇 buffer[skip_window]，label 是 buffer[target]。同时，因为这个语境单词被使用了，所以再把它添加到 targets_to_avoid 中过滤。在对一个目标单词生生成完所有样本后（num_skips 个样本），我们再读入下一个单词（同时会抛掉 buffer 中第一个单词），即把滑窗向后移动一位，这样我们的目标单词也向后移动了一个，语境单词也整体后移了，便可以开始生成下一个目标单词的训练样本。

两层循环完成后，我们已经获得了 batch_size 个训练样本，将 batch 和 labels 作为函数结果返回。

```python
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
```

到目前为止，我们对训练数据的生成完成，接下来实现 Word2Vec。

### 实现 Word2Vec

#### 定义训练时需要的参数

```python
batch_size = 128
embedding_size = 128  # 将单词转为稠密向量的维度，一般是500~1000这个范围内的值，这里设为128
skip_window = 1  # 单词间最远可以联系到的距离
num_skips = 2  # 对每个目标单词提取的样本数

# 生成验证数据，随机抽取一些频数最高的单词，看向量空间上跟它们距离最近的单词是否相关性比较高
valid_size = 16  # 抽取的验证单词数
valid_window = 100  # 验证单词只从频数最高的 100 个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 随机抽取
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量
```

#### 定义 Skip-Gram Word2Vec 模型网络结构

Skip-Gram模型有两个输入。一个是一组用整型表示的上下文单词，另一个是目标单词。给这些输入建立占位符节点，之后就可以填入数据了。

```python
# 建立输入占位符
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  # 将前面随机产生的 valid_examples 转为 TensorFlow 中的 constant
```

这里谈得都是嵌套，那么需要定义一个嵌套参数矩阵。我们用唯一的随机值来初始化这个大矩阵。

```python
# 随机生成所有单词的词向量 embeddings，单词表大小 5000，向量维度 128
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        
```

对噪声-比对的损失计算就使用一个逻辑回归模型。对此，我们需要对语料库中的每个单词定义一个权重值和偏差值。(也可称之为`输出权重` 与之对应的 `输入嵌套值`)。定义如下。

```python
nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_bias = tf.Variable(tf.zeros([vocabulary_size]))
```

然后我们需要对批数据中的单词建立嵌套向量，TensorFlow提供了方便的工具函数。

```python
# 查找 train_inputs 对应的向量 embed
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```

现在我们有了每个单词的嵌套向量，接下来就是使用噪声-比对的训练方式来预测目标单词。

```python
loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=train_labels, inputs=embed, num_sampled=num_sampled,
                       num_classes=vocabulary_size))
```

我们对损失函数建立了图形节点，然后我们需要计算相应梯度和更新参数的节点，比如说在这里我们会使用随机梯度下降法，TensorFlow也已经封装好了该过程。

```python
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
```

#### 训练模型

定义最大的迭代次数为 10 万次，然后创建并设置默认的 session，并执行参数和初始化。在每一步迭代中，先使用generate_batch 生成一个 batch 的 inputs 和 labels 数据，并用他们创建 feed_dict。然后使用 session.run() 执行一次优化器运算（即一次参数更新）和损失计算，并将这一步训练的loss 积累到 average_loss。

之后每2000 次循环，计算一个平均 loss 并显示出来。

每 10000 次循环，计算一次验证单词与全部单词的相似度，并将每个验证单词最相近的 8 个单词显示出来。

```python
with tf.Session(graph=graph) as session:
    init.run()
    print 'Initialized'

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print 'Average loss at step {} : {}'.format(step, average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to {} :'.format(valid_word)

                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '{} {},'.format(log_str, close_word)
                print log_str
        final_embeddings = normalized_embeddings.eval()s
```

#### 结果可视化

下面定义一个用来可视化 Word2Vec 效果的函数。这里 low_dim_embs 是降维到 2 维 的单词的空间向量，我们将在图表中展示每个单词的位置。我么使用 plt.scatter 显示散点图（单词的位置），并用 plt.annotate 展示单词本身，同时，使用 plt.savefig 保存图片到本地文件。

```python
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels then embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)
```

我们使用 sklearn.manifold.TSNE 实现降维，这里直接将原始的 128 维的嵌入向量降到 2 维，再用前面的 plot_with_labels 函数进行展示。这里只展示词频最高的 100 个单词的可视化结果。

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
```

从可视化结果可以看粗，距离相近的单词在语义上具有很高的相似性。在训练 Word2Vec 模型时，为了获得比较好的结构，我们可以使用大规模的语料库，同时需要对参数进行调试，选取最合适的值。