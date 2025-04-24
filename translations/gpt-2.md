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

### 那么，语言模型到底是什么？

#### 什么是语言模型

在[图解 Word2Vec](https://jalammar.github.io/illustrated-word2vec/)一文中，我们已经探讨过语言模型的概念 —— 本质上，它是一个机器学习模型，能够根据一句话中的一部分内容预测下一个词语。

最广为人知的语言模型例子就是我们手机上的键盘，它们会根据你目前输入的内容预测下一个你可能会输入的词。

![示例图 - 手机键盘自动补全](https://jalammar.github.io/images/word2vec/swiftkey-keyboard.png)

从这个角度来看，我们可以说 GPT-2 基本上就是一个“下一个词预测”功能，类似于手机键盘应用里的功能——只不过 GPT-2 的规模远远大于手机上的模型，也复杂得多。

GPT-2 是在一个名为 WebText 的庞大数据集上训练出来的，这个数据集容量高达 40GB，是 OpenAI 的研究人员从互联网上抓取而来，用于研究目的。

为了做个对比：我所使用的键盘应用 SwiftKey 占用的空间是 78MB。而 GPT-2 最小的变体模型，仅其参数就需要 500MB 来存储。最大的 GPT-2 模型大小是它的 13 倍，可能需要超过 6.5GB 的存储空间。

![示例图 - GPT-2模型类型](https://jalammar.github.io/images/gpt2/gpt2-sizes.png)