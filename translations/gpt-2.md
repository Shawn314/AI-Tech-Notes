# 图解 GPT-2（Transformer 语言模型的可视化解析）

*作者：Jay Alammar | 原文链接：[https://jalammar.github.io/illustrated-gpt2](https://jalammar.github.io/illustrated-gpt2)*

今年，我们见证了机器学习领域的一次惊艳应用。OpenAI 的 GPT-2 展现出撰写连贯而富有激情的文章的惊人能力，超出了我们对当今语言模型的预期。GPT-2 本身的架构并不是特别新颖 —— 它的架构与仅使用解码器（decoder-only）的 Transformer 非常相似。但 GPT-2 是一个非常庞大的、基于 Transformer 的语言模型，训练数据集也异常庞大。

在这篇文章中，我们将深入了解支撑该模型取得如此成果的架构。我们将深入探讨其自注意力（self-attention）层的工作机制。随后，我们还会看看仅使用解码器的 Transformer 在语言建模之外的应用。

这篇文章的另一个目标是补充我之前的文章《图解 Transformer》，通过更多的视觉图示解释 Transformer 的内部工作原理，以及它们自原始论文发表以来的演化。我希望这种视觉化语言能帮助大家更轻松地理解后续不断演变的基于 Transformer 的模型。
## 目录

### 第一部分：GPT-2 与语言建模
- 什么是语言模型（Language Model）
- Transformer 在语言建模中的应用
- 与 BERT 的一个区别
- Transformer 模块的演化
- 脑外科速成课：深入了解 GPT-2
- 更深入的剖析
- 第一部分结语：GPT-2，女士们先生们

### 第二部分：图解自注意力机制
- 自注意力机制（无掩码）
  1. 创建查询（Query）、键（Key）和值（Value）向量
  2. 打分（Score）
  3. 加权求和（Sum）
- 图解掩码自注意力机制
- GPT-2 的掩码自注意力
- 超越语言建模
- 你成功了！

### 第三部分：语言建模之外
- 机器翻译
- 文本摘要
- 迁移学习
- 音乐生成

## 第一部分：GPT-2 与语言建模

那么，语言模型到底是什么？

### 什么是语言模型

在[图解 Word2Vec](https://jalammar.github.io/illustrated-word2vec/)一文中，我们已经探讨过语言模型的概念 —— 本质上，它是一个机器学习模型，能够根据一句话中的一部分内容预测下一个词语。

最广为人知的语言模型例子就是我们手机上的键盘，它们会根据你目前输入的内容预测下一个你可能会输入的词。

![示例图 - 手机键盘自动补全](https://jalammar.github.io/images/word2vec/swiftkey-keyboard.png)

从这个角度来看，我们可以说 GPT-2 基本上就是一个“下一个词预测”功能，类似于手机键盘应用里的功能——只不过 GPT-2 的规模远远大于手机上的模型，也复杂得多。

GPT-2 是在一个名为 WebText 的庞大数据集上训练出来的，这个数据集容量高达 40GB，是 OpenAI 的研究人员从互联网上抓取而来，用于研究目的。

为了做个对比：我所使用的键盘应用 SwiftKey 占用的空间是 78MB。而 GPT-2 最小的变体模型，仅其参数就需要 500MB 来存储。最大的 GPT-2 模型大小是它的 13 倍，可能需要超过 6.5GB 的存储空间。

![示例图 - GPT-2模型类型](https://jalammar.github.io/images/gpt2/gpt2-sizes.png)

一种很好的实验GPT-2的方法是使用AllenAI的GPT-2 Explorer。它使用GPT-2展示下一个单词的十个可能预测（以及它们的概率分数）。你可以选择一个单词，然后查看接下来的预测列表，以继续写作这段文字。

### Transformer 在语言建模中的应用

正如我们在[插图版Transformer](https://jalammar.github.io/illustrated-transformer/)中看到的，原始的Transformer模型由编码器和解码器组成——每个部分都是一个堆叠的Transformer模块。这个架构是合适的，因为该模型解决的是机器翻译问题——这是一个在过去，编码器-解码器架构成功的领域。

![](https://jalammar.github.io/images/xlnet/transformer-encoder-decoder.png)

许多后续的研究工作使得该架构去除了编码器或解码器，只使用了一个堆叠的Transformer模块——将其堆叠得尽可能高，输入大量的训练文本，并投入巨大的计算资源（训练这些语言模型的成本可能高达数十万美元，对于AlphaStar这样的模型，可能需要数百万美元）。

![](https://jalammar.github.io/images/gpt2/gpt-2-transformer-xl-bert-3.png)

我们可以将这些模块堆叠得多高呢？事实证明，这正是不同GPT-2模型大小之间的主要区别之一：

![](https://jalammar.github.io/images/gpt2/gpt2-sizes-hyperparameters-3.png)

### 来自BERT的一个区别

> **First Law of Robotics**, 
> A robot may not injure a human being or, through inaction, allow a human being to come to harm.


GPT-2是通过Transformer解码器模块构建的。而BERT则使用了Transformer编码器模块。我们将在后续的章节中探讨这两者之间的区别。但两者之间的一个关键区别是，GPT-2像传统语言模型一样，一次输出一个token。例如，让我们通过一个提示语来让一个训练好的GPT-2背诵机器人法则第一条：

![](https://jalammar.github.io/images/xlnet/gpt-2-output.gif)

这些模型的实际工作方式是，在每个token被生成后，该token会被添加到输入序列中。然后，这个新的序列会成为模型在下一步的输入。这种思想叫做“自回归（auto-regression）”。这是使得RNNs异常有效的一个思想。

![](https://jalammar.github.io/images/xlnet/gpt-2-autoregression-2.gif)

GPT-2以及一些后续模型，如TransformerXL和XLNet，天生是自回归的。而BERT则不是。这是一个权衡。在失去自回归的同时，BERT获得了能够结合单词两侧上下文的能力，从而获得更好的结果。XLNet则在恢复自回归的同时，找到了一种替代方法来结合单词两侧的上下文。

### Transformer模块的演变

[最初的Transformer论文](https://arxiv.org/abs/1706.03762)介绍了两种类型的Transformer模块：

#### 编码器模块
首先是编码器模块：

![](https://jalammar.github.io/images/xlnet/transformer-encoder-block-2.png)
> 来自原始Transformer论文的编码器模块可以处理直到某个最大序列长度的输入（例如512个token）。如果输入序列短于此限制，也没问题，我们只需对剩余的序列进行填充。

#### 解码器模块
其次是解码器模块，它与编码器模块有一些小的架构差异——增加了一层，使其能够关注编码器的特定部分：

这里自注意力层的一个关键区别是，它会屏蔽未来的token——不是像BERT那样将词语改为[mask]，而是在自注意力计算中通过干扰阻止来自计算位置右侧的token信息。

例如，如果我们要突出显示位置#4的路径，我们可以看到它仅被允许关注当前和之前的token：

![](https://jalammar.github.io/images/xlnet/transformer-decoder-block-self-attention-2.png)

明确区分自注意力（BERT使用）和掩蔽自注意力（GPT-2使用）非常重要。正常的自注意力模块允许一个位置查看它右侧的token。而掩蔽自注意力则阻止这种情况的发生：

![](https://jalammar.github.io/images/gpt2/self-attention-and-masked-self-attention.png)

#### 仅解码器模块
在原始论文之后，生成维基百科的长序列摘要模型提出了另一种Transformer模块的安排，能够进行语言建模。该模型抛弃了Transformer编码器。因此，我们称这种模型为“Transformer-Decoder”。这个早期的基于Transformer的语言模型由六个Transformer解码器模块堆叠而成：

![](https://jalammar.github.io/images/xlnet/transformer-decoder-intro.png)

> 这些解码器模块是相同的。我已经扩展了第一个模块，以便你可以看到它的自注意力层是掩蔽版本。请注意，该模型现在可以在某一段中处理最多4,000个token——相比原始Transformer中的512个，这是一次巨大的升级。

这些模块与原始的解码器模块非常相似，只是去掉了第二个自注意力层。在[《使用更深自注意力进行字符级语言建模》](https://arxiv.org/pdf/1808.04444.pdf)一文中，研究人员使用了类似的架构，创建了一个一次预测一个字母/字符的语言模型。

OpenAI的GPT-2模型使用的正是这些仅解码器模块。

### 脑外科速成课程：深入探究GPT-2

> Look inside and you will see, The words are cutting deep inside my brain. Thunder burning, quickly burning, Knife of words is driving me insane, insane yeah. ~Budgie

让我们把一个训练好的GPT-2放在手术台上，看看它是如何工作的。

![](https://jalammar.github.io/images/gpt2/gpt-2-layers-2.png)
> GPT-2可以处理1024个token。每个token沿着自己的路径穿过所有解码器模块。

运行训练好的GPT-2最简单的方法是让它自由发挥（这在技术上叫做生成无条件样本）——或者，我们可以给它一个提示，让它谈论某个话题（即生成交互式条件样本）。在自由发挥的情况下，我们只需提供开始token，让它开始生成词语（训练好的模型使用<|endoftext|>作为起始token。我们这里用\<s>代替）。

![](https://jalammar.github.io/images/gpt2/gpt2-simple-output-2.gif)

模型只有一个输入token，因此只有这一路径是活跃的。这个token依次通过所有层，然后沿着这条路径生成一个向量。这个向量可以与模型的词汇表（模型知道的所有单词，GPT-2的词汇表包含50,000个单词）进行比对。在这个例子中，我们选择了概率最高的token——‘The’。但是我们当然可以稍作变化——你知道，当你在键盘应用程序中不断点击建议的词语时，有时它会陷入重复的循环，唯一的解决方法就是点击第二个或第三个建议的词语。在这里也会发生同样的情况。GPT-2有一个名为top-k的参数，我们可以使用它让模型考虑除最高概率词以外的其他词语（当top-k=1时，就是只选择最高词语的情况）。

![](https://jalammar.github.io/images/gpt2/gpt2-simple-output-2.gif)

在下一步中，我们将第一步的输出添加到输入序列中，并让模型进行下一次预测：

![](https://jalammar.github.io/images/gpt2/gpt-2-simple-output-3.gif)

请注意，在这个计算中，第二条路径是唯一活跃的。GPT-2的每一层都保留了对第一个token的解释，并将在处理第二个token时使用它（我们将在接下来的自注意力部分详细讲解）。GPT-2不会根据第二个token重新解释第一个token。

### 更深入的探究

#### 输入编码

让我们更详细地了解模型，开始从输入部分。与我们之前讨论过的其他NLP模型一样，模型在其嵌入矩阵中查找输入词的嵌入——这是我们作为训练模型一部分得到的组件之一。

![](https://jalammar.github.io/images/gpt2/gpt2-token-embeddings-wte-2.png)

> 每一行是一个词嵌入：一个表示词语并捕捉其某些意义的数字列表。这个列表的大小在不同的GPT-2模型大小中是不同的。最小的模型每个词的嵌入大小为768。

因此，一开始，我们会在嵌入矩阵中查找开始Token \<s> 的嵌入。在将其传递给模型的第一个模块之前，我们需要加入位置编码——这是一个信号，用来指示词语在序列中的顺序。训练模型的一部分是一个矩阵，包含了输入中每个位置（总共有1024个位置）的位置编码向量。

![](https://jalammar.github.io/images/gpt2/gpt2-positional-encoding.png)

到此为止，我们已经了解了输入词语在传递给第一个Transformer模块之前是如何处理的。我们还知道了构成训练好的GPT-2的两个权重矩阵。

![](https://jalammar.github.io/images/gpt2/gpt2-input-embedding-positional-encoding-3.png)

> 将一个词传递给第一个Transformer模块意味着查找它的嵌入，并将位置#1的位置信息编码向量加上。

#### 沿着堆栈的旅程

现在，第一个模块可以处理这个Token，首先将它通过自注意力过程，然后传递给其神经网络层。一旦第一个Transformer模块处理完这个Token，它会将其结果向量传递到堆栈的下一个模块进行处理。每个模块的处理过程是相同的，但每个模块在自注意力和神经网络子层中都有自己的权重。

![](https://jalammar.github.io/images/gpt2/gpt2-transformer-block-vectors-2.png)

#### 自注意力回顾

语言在很大程度上依赖于上下文。例如，看看机器人法则的第二条：

> **Second Law of Robotics**  
> A robot must obey the orders given <font color="#FF0000">it</font> by human beings except where <font color="#00FF00">such orders</font> would conflict with the <font color="#808080">First Law</font>.

我在句子中标出了三个地方，词语指代了其他词语。没有上下文的支持，我们无法理解或处理这些词。当模型处理这句话时，它必须能够知道：

- <font color="#FF0000">it</font>指的是“robot”
- <font color="#00FF00">such orders</font>指的是法则的前半部分，即“the orders given it by human beings”
- <font color="#808080">First Law</font>指的是完整的第一法则

这就是自注意力的作用。它将模型对相关和关联词的理解融入其中，在处理一个词之前，就先处理这些词的上下文（通过神经网络）。它通过为每个词在片段中的相关性打分，然后将这些词的向量表示加总起来。

例如，在处理“it”这个词时，顶部模块的自注意力层正关注“a robot”。它将传递给神经网络的向量是这三个词的向量之和，乘以它们的得分。

![](https://jalammar.github.io/images/gpt2/gpt2-self-attention-example-2.png)

#### 自注意力过程

自注意力在每个Token的路径上进行处理。重要的组成部分是三个向量：

- **查询（Query）**：查询是当前词的表示，用来与所有其他词的表示（键）进行匹配。我们只关心当前处理Token的查询向量。
- **键（Key）**：键向量类似于所有词的标签。它们是我们在寻找相关词时进行匹配的对象。
- **值（Value）**：值向量是实际的词表示。一旦我们为每个词计算了其相关性得分，这些值就是我们加总起来，表示当前词的内容。
  
![](https://jalammar.github.io/images/gpt2/self-attention-example-folders-3.png)

一个粗略的类比是，将其想象成在文件柜中查找信息。查询就像是你写下的研究主题的便签，键就像是柜子里文件夹的标签。当你将标签和便签匹配时，你就取出该文件夹的内容，这些内容就是值向量。只不过你并不是只在寻找一个值，而是从多个文件夹中混合多个值。将查询向量与每个键向量相乘，得到每个文件夹的得分（技术上是：点积，然后是softmax）。

![](https://jalammar.github.io/images/gpt2/self-attention-example-folders-scores-3.png)

我们将每个值与其得分相乘并求和——得出我们的自注意力结果。

![](https://jalammar.github.io/images/gpt2/gpt2-value-vector-sum.png)

这种加权混合的值向量产生了一个向量，其中50%的“注意力”集中在“robot”一词，30%集中在“a”一词，19%集中在“it”一词。稍后我们将深入探讨自注意力的更多细节。但首先，让我们继续向堆栈的顶部推进，直到模型输出。

#### 模型输出

当模型的顶部模块产生它的输出向量（即它自己的自注意力和神经网络结果）时，模型将该向量与嵌入矩阵相乘。

![](https://jalammar.github.io/images/gpt2/gpt2-output-projection-2.png)

回想一下，嵌入矩阵中的每一行对应于模型词汇表中一个词的嵌入。这个乘法的结果被解释为每个词在模型词汇表中的得分。

![](https://jalammar.github.io/images/gpt2/gpt2-output-scores-2.png)

我们可以简单地选择得分最高的token（top_k=1）。但如果模型同时考虑其他词，效果会更好。因此，一个更好的策略是从整个词汇表中根据得分选取一个词，将得分作为选择该词的概率（得分越高，被选中的概率越大）。一个折衷的做法是将top_k设置为40，让模型考虑得分最高的40个词。

![](https://jalammar.github.io/images/gpt2/gpt2-output.png)

至此，模型完成了一次迭代，并输出了一个词。模型会继续迭代，直到生成完整的上下文（1024个token）或直到生成结束标记（end-of-sequence token）。


#### 第1部分结束：GPT-2，女士们先生们

到这里，我们了解了GPT-2的工作原理。如果你对自注意力层内部的具体细节感兴趣，那么接下来的奖励部分将为你提供更多信息。我创建了这部分内容，用来引入更多的可视化语言来描述自注意力，以便后续的Transformer模型（如TransformerXL和XLNet）能够更容易被检查和描述。

我想指出几处在这篇文章中的简化处理：

- 我将“词”和“Token”交替使用。但实际上，GPT-2使用字节对编码（Byte Pair Encoding）来创建其词汇中的token。这意味着这些token通常是词的一部分。
- 我们展示的例子运行的是GPT-2的推理/评估模式。这就是为什么它每次只处理一个词。在训练时，模型会针对更长的文本序列进行训练，一次处理多个token。同时在训练时，模型会处理更大的批量（512）而非评估时使用的批量大小1。
- 在旋转/转置向量时，我做了一些简化，以便更好地管理图像中的空间。在实现时，必须更加精确。
- Transformers使用了大量的层归一化，这非常重要。我们在《插图Transformer》中提到过其中的一些，但在本文中我们更侧重于自注意力。
- 有时我需要显示更多的框来表示一个向量，我将这些表示为“放大”。例如：
  
![](https://jalammar.github.io/images/gpt2/zoom-in.png)


## 第二部分：图解自注意力机制

在本文前面，我们展示了这样一张图片，展示了自注意力（Self-Attention）如何应用在处理单词 "it" 的一层中：

![](https://jalammar.github.io/images/gpt2/gpt2-self-attention-1-2.png)

在本节中，我们将详细探讨这一过程是如何实现的。  
需要注意的是，我们会以一种能够理解每个单词变化的方式来讲解。  
因此，你会看到很多单个向量的示意图。  
实际上，真实的实现过程是通过巨大的矩阵相乘来完成的，但这里我们更关注每个单词层面的直观理解。

### 自注意力机制（无掩码）

我们先从编码器模块中最初的自注意力计算方式开始讲起。  
假设我们有一个简化版的小型 Transformer 块，每次只能处理 4 个 token。

自注意力的计算主要分为三个步骤：

1. 为每个路径创建查询（Query）、键（Key）、值（Value）向量。
2. 对于每个输入的 token，使用它的查询向量去与所有其他的键向量进行打分。
3. 将所有值向量乘以对应的分数后相加，得到最终结果。

![](https://jalammar.github.io/images/xlnet/self-attention-summary.png)

#### 1 - 创建查询（Query）、键（Key）、值（Value）向量

让我们专注于第一个路径（token）。  
我们将提取它的查询向量，并将其与所有键向量进行对比，从而为每个键生成一个得分（Score）。  
自注意力的第一步，就是为每一个 token 路径计算出三种向量（这里我们暂时忽略多头注意力机制）：

![](https://jalammar.github.io/images/xlnet/self-attention-1.png)

#### 2 - 打分（Score）

现在我们已经有了这些向量，第二步只需要用查询向量和键向量。  
由于我们关注的是第一个 token，我们用它的查询向量去分别与四个 token 的键向量进行点积运算，从而为每个 token 计算出一个得分。

![](https://jalammar.github.io/images/xlnet/self-attention-2.png)

#### 3 - 加权求和（Sum）

接下来，我们将这些得分与对应的值向量相乘。  
得分越高的值向量，在最终相加得到的结果向量中所占的比例也越大。

![alt text](https://jalammar.github.io/images/xlnet/self-attention-3-2.png)

> 为了直观展示：得分越低的值向量，我们会以更透明的方式呈现，表示它们被乘以一个很小的数后对结果的贡献较小。 

如果我们对每一条路径都进行相同的操作，最终就能为每个 token 得到一个新的向量，  
这个向量包含了该 token 所需的上下文信息。

随后，这些向量将被送入 Transformer 块的下一个子层 —— 前馈神经网络（Feed-Forward Neural Network）。

![alt text](https://jalammar.github.io/images/xlnet/self-attention-summary.png)

### 图解掩码自注意力机制（Masked Self-Attention）

在了解了 Transformer 中自注意力（Self-Attention）的工作机制之后，  
现在我们来看看掩码自注意力（Masked Self-Attention）。

掩码自注意力与普通自注意力基本相同，只是在第 2 步（打分）时有所不同。  
假设模型的输入中只有两个 token，而我们正在处理第二个 token。  
在这种情况下，未来的 token 会被屏蔽（masked）。  
也就是说，在打分步骤中，模型会干预，使得对未来 token 的打分总是 0，  
这样模型就无法偷看未来的单词：

![alt text](https://jalammar.github.io/images/xlnet/masked-self-attention-2.png)

通常，这种屏蔽是通过一个叫做 **注意力掩码（attention mask）** 的矩阵来实现的。  
想象一组四个单词组成的序列（比如 “robot must obey orders”）。  
在语言建模任务中，这个序列会被分四步输入模型 —— 每步输入一个单词（这里假设每个单词对应一个 token）。

![alt text](https://jalammar.github.io/images/gpt2/transformer-decoder-attention-mask-dataset.png)

由于这些模型是按批次（batch）工作的，  
我们可以假设这个小型示例的 batch size 为 4，  
即一次性处理完整的四步序列。

在矩阵运算中，得分是通过将查询矩阵（queries matrix）与键矩阵（keys matrix）相乘得到的。  
可以将它想象为一个表格，只不过每个单元格里放的是对应单词的查询或键向量：

![alt text](https://jalammar.github.io/images/gpt2/queries-keys-attention-mask.png)

完成矩阵相乘后，我们应用一个掩码三角形（attention mask triangle）。  
掩码会将我们不想看到的位置设为负无穷（或者非常大的负数，比如在 GPT-2 中是 -10^9）。

![alt text](https://jalammar.github.io/images/gpt2/transformer-attention-mask.png)

然后，在每一行应用 softmax 函数，得到真正用于自注意力的得分（scores）：

![alt text](https://jalammar.github.io/images/gpt2/transformer-attention-masked-scores-softmax.png)

这些得分表的含义如下：

- 当模型处理第一个样本（第 1 行，即单词 “robot”）时，  
  100% 的注意力会集中在这个单词上。
- 当模型处理第二个样本（第 2 行，即单词序列 “robot must”）时，  
  处理单词 “must” 时，48% 的注意力放在 “robot”，52% 的注意力放在 “must”。
- 以此类推。
