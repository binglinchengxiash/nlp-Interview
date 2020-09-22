# Bert

### 1、BERT的基本原理是什么？

BERT可以看成一个自编码的语言模型，主要用两个任务训练该模型:

MLM\(Masked LM\)

输入一句话的时候，随机地选一些要预测的词，然后用一个特殊的符号\[MASK\]来代替它们，之后让模型根据所给的标签去学习这些地方该填的词

NSP\(Next Sentence Prediction\)

在双向语言模型的基础上额外增加了一个句子级别的连续性预测任务，即预测输入BERT的两段文本是否为连续的文本

### 2. BERT是怎么用Transformer的？

BERT只使用了Transformer的Encoder模块,与Transformer本身的Encoder端相比，BERT的Transformer Encoder端输入的向量表示，多了Segment Embeddings。

### 3. 请简要介绍一下Masked LM

MLM可以看作是一种引入噪声的手段，增强模型的泛化能力

具体步骤：

如果某个Token在被选中的15%个Token里，则按照下面的方式随机的执行：

80%的概率替换成\[MASK\]，比如my dog is hairy → my dog is \[MASK\]

10%的概率替换成随机的一个词，比如my dog is hairy → my dog is apple

10%的概率替换成它本身，比如my dog is hairy → my dog is hairy

好处：

BERT并不知道\[MASK\]替换的是这15%个Token中的哪一个词\(意思是输入的时候不知道\[MASK\]替换的是哪一个词，但是输出还是知道要预测哪个词的\)，而且任何一个词都有可能是被替换掉的，比如它看到的apple可能是被替换的词。这样强迫模型在编码当前时刻的时候不能太依赖于当前的词，而要考虑它的上下文，甚至对其上下文进行”纠错”。比如上面的例子模型在编码apple是根据上下文my dog is应该把apple\(部分\)编码成hairy的语义而不是apple的语义。

### 4. 请简要介绍一下NSP\(Next Sentence Prediction\)

判断第二句话在文本中是否紧跟在第一句话之后，类似于段落排序

将一篇文章的各段打乱，通过重新排序把原文还原出来，这其实需要对全文大意有充分、准确的理解。Next Sentence Prediction 任务实际上就是段落重排序的简化版：只考虑两句话，判断是否是一篇文章中的前后句。在实际预训练过程中，作者从文本语料库中随机选择 50% 正确语句对和 50% 错误语句对进行训练，与 Masked LM 任务相结合，让模型能够更准确地刻画语句乃至篇章层面的语义信息。

### 5. EMO、GPT、BERT三者之间有什么区别？

特征提取器：

ELMO采用LSTM进行提取，GPT和BERT则采用Transformer进行提取。很多任务表明Transformer特征提取能力强于LSTM

ELMO用1层静态向量+2层LSTM，多层提取能力有限，而GPT和BERT中的Transformer可采用多层，并行计算能力强。

单/双向语言模型：

GPT采用单向语言模型，EMLo和BERT采用双向语言模型。但是EMLo实际上是两个单向语言模型（方向相反）的拼接，这种融合特征的能力比BERT一体化融合特征方式弱。

GPT和BERT都采用Transformer，Transformer是encoder-decoder结构，GPT的单向语言模型采用decoder部分，decoder的部分见到的都是不完整的句子；bert的双向语言模型则采用encoder部分，采用了完整句子

### 6. BERT的输入和输出分别是什么？

bert的输入

bert的输入是三部分，input\_id segment\_id 和position\_id  
 input\_id 就是将输入转换成词表中的id，通过查询字向量表将文本中的每个字转换为一维向量，也可以利用Word2Vector等算法进行预训练以作为初始值；  
 segment\_id（token\_type\_id）就是区分句子是第几个句子 用0和1表示  
 position\_id是记录句子中词的顺序

  
 形式：\[cls\]上一句话，\[sep\]下一句话.\[sep\]

bert的输出

bert的输出有两种，一种是get\_sequence\_out\(\),获取的是整个句子每一个token的向量表示，输出shape是\[batch\_size, seq\_length, hidden\_size\]；  
 另一种是get\_pooled\_out\(\)，获取的是整个句子中的\[cls\]的表示,输出shape是\[batch size,hidden size\]。

### 7. Mask LM相对于CBOW有什么异同点？

相同点：

都是根据上下文预测中心词

不同点：

CBOW中的输入数据只有待预测单词的上下文，而BERT的输入是带有\*\*\[MASK\] token\*\*的“完整”句子

CBOW模型训练后，每个单词的word embedding是唯一的，因此并不能很好的处理一词多义的问题，而BERT模型得到的word embedding\(token embedding\)融合了上下文的信息，就算是同一个单词，在不同的上下文环境下，得到的word embedding是不一样的。

Word2vec的inference就是对该矩阵的embedding\_lookup，对同一个词得到的词向量肯定是不变的，所以是静态的。

而bert的inference过程会复杂很多，会利用到上下文的信息经过transformer编码（即self-attention交互），同一个词如果上下文信息不同，那么就会得到不同的embedding，所以是动态的。

### 8. BERT的两个预训练任务对应的损失函数是什么

![](file:///C:/Users/songhuan/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

### 9. BERT优缺点

优点  
 bert将双向 Transformer 用于语言模型，Transformer 的 encoder 是一次性读取整个文本序列，而不是从左到右或从右到左地按顺序读取，这个特征使得模型能够基于单词的两侧学习，相当于是一个双向的功能。

bert 相较于rnn,lstm可以并发进行，并且可以提取不同层次的信息,反应更全面的句子语义。相较于 word2vec，根据句子上下文获取词义，避免了歧义出现。

缺点  
 bert模型参数太多，而且模型太大，少量数据训练时，容易发生过拟合。  
 bert模型预训练会出现mask,而在下游的微调任务中并不会出现，使得不匹配。  
 bert模型会将词分成词根一样的词片，若随机遮掩一些词的时候，若遮掩住中间的的词片，则会发生不是根据上下文的语义预测的。

### 10、简述 bert

Bert 是一个预训练语言模型，它主要有两个任务。第一个任务是将数据集中的句子随机遮掩一部分，通过训练，来预测这些词是什么，加强了句子内部之间的联系；第二个任务是判断两个句子是否是连续的上下句，通过训练来识别，加强了句子外部之间的联系。

bert 的创新点在于它将双向 Transformer 用于语言模型，Transformer 的 encoder 是一次性读取整个文本序列，而不是从左到右或从右到左地按顺序读取，这个特征使得模型能够基于单词的两侧学习，相当于是一个双向的功能。

bert 相较于rnn,lstm可以并行进行，并且可以提取不同层次的信息,反应更全面的句子语义。相较于 word2vec，根据句子上下文获取词义，避免了歧义出现。缺点就是模型参数太多，而且模型太大，少量数据训练时，容易发生过拟合。

### 11、bert的模型结构

bert 是由transformer的编码器构成的。  
 小Bert是由12个transformer的编码器组成，大bert是由24个transformer的编码器组成，同时相较于transformer的输入而言，bert的输入多了segment id。

### 12、Multi-head Attention 多头注意力机制

![IMG\_256](file:///C:/Users/songhuan/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)

多头注意力机制定义一个超参数h\(head数量\)来提取多重语义，先将Query，Key，Value做线性变化，然后把Query，Key，Value的embedding维度分割成了h份，q与k做相似度计算，然后与v做加权求和得到注意力向量，再把这些注意力向量拼接起来，再通过线性变换得到最终的结果。注意头之间参数不共享，每次Q，K，V进行线性变换的参数是不一样的。  
  


### 13、bert适合的任务

bert可以用来做分类、句子语义相似度、阅读理解等任务  
 对于分类：获得输出的\[cls\]的向量，做softmax得到每个类别的概率  
 对于语义相似度：获得输出的\[cls\]的向量，做softmax得到相似度  
 对于阅读理解：输入\[cls\]问题\[sep\]文本\[sep\]，输出的是整句话的向量，将其分成两部分，转换成对应的答案开始和结束位置的概率。

### 14、Bert中如何获得词意和句意？

get\_pooled\_out代表了涵盖了整条语句的信息    cls向量  \(bacth\_size\*hidden\_size\)

get\_sentence\_out代表了这个获取每个token的output 输出  \(bacth\_size\*sen\_len\*hidden\_size\)

### 15、源码中Attention后实际的流程是如何的？

Transform模块中：在残差连接之前，对output\_layer进行了dense+dropout后再合并input\_layer进行的layer\_norm得到的attention\_output

所有attention\_output得到并合并后，也是先进行了全连接，而后再进行了dense+dropout再合并的attention\_output之后才进行layer\_norm得到最终的layer\_output

### 16、为什么要在Attention后使用残差结构？

残差结构能够很好的消除层数加深所带来的信息损失问题

### 17、平时用官方Bert包么？耗时怎么样？

第三方：bert\_serving

官方：bert\_base

耗时：64GTesla，64max\_seq\_length，80-90doc/s

在线预测只能一条一条的入参，实际上在可承受的计算量内batch越大整体的计算性能性价比越高

### 18、Bert的双向体现在什么地方？

mask+attention，mask的word结合全部其他encoder word的信息

### 19、为什么选取的15%的词中有80%用\[mask\]来替换，10%用原词，剩余的10%用随机词来替换，不全部用mask来遮掩:

是因为下游任务微调中不会出现mask，这样会导致预训练模型和下游任务不匹配。同时在预测时，因为不知道这个词是否是正确的，会使模型更加依赖上下文，有一定的纠错能力。

=====另一版本======

mask只会出现在构造句子中，当真实场景下是不会出现mask的，全mask不match句型了

随机替换也帮助训练修正了\[unused\]和\[UNK\]；强迫文本记忆上下文信息

### 20、为什么BERT有3个嵌入层，它们都是如何实现的？

input\_id是语义表达，和传统的w2v一样，方法也一样的lookup

segment\_id是辅助BERT区别句子对中的两个句子的向量表示，从\[1,embedding\_size\]里面lookup

position\_id是为了获取文本天生的有序信息，否则就和传统词袋模型一样了，从\[511,embedding\_size\]里面lookup

### 21、bert的具体网络结构，以及训练过程，bert为什么火，它在什么的基础上改进了些什么？

bert是用了transformer的encoder侧的网络，作为一个文本编码器，使用大规模数据进行预训练，预训练使用两个loss，一个是mask LM，遮蔽掉源端的一些字（可能会被问到mask的具体做法，15%概率mask词，这其中80%用\[mask\]替换，10%随机替换一个其他字，10%不替换，至于为什么这么做，那就得问问BERT的作者了\{捂脸}），然后根据上下文去预测这些字，一个是next sentence，判断两个句子是否在文章中互为上下句，然后使用了大规模的语料去预训练。在它之前是GPT，GPT是一个单向语言模型的预训练过程（它和gpt的区别就是bert为啥叫双向 bi-directional），更适用于文本生成，通过前文去预测当前的字。下图为transformer的结构，bert的网络结构则用了左边的encoder。

### 22、讲讲multi-head attention的具体结构

BERT由12层transformer layer（encoder端）构成，首先word emb , pos emb（可能会被问到有哪几种position embedding的方式，bert是使用的哪种）, sent emb做加和作为网络输入，每层由一个multi-head attention, 一个feed forward 以及两层layerNorm构成，一般会被问到multi-head attention的结构，具体可以描述为，一个768的hidden向量，被映射成query， key， value。 然后三个向量分别切分成12个小的64维的向量，每一组小向量之间做attention。

hidden\(768\) -&gt; query\(768\) -&gt; 12 x 64

hidden\(768\) -&gt; key\(768\) -&gt; 12 x 64

hidden\(768\) -&gt; val\(768\) -&gt; 12 x 64

然后query和key之间做attention，得到一个12乘以12的权重矩阵，然后根据这个权重矩阵加权val中切分好的12个64维向量，得到一个12 x 64的向量，拉平输出为768向量。

问题2.5: Bert 采用哪种Normalization结构，LayerNorm和BatchNorm区别，LayerNorm结构有参数吗，参数的作用？

采用LayerNorm结构，和BatchNorm的区别主要是做规范化的维度不同，BatchNorm针对一个batch里面的数据进行规范化，针对单个神经元进行，比如batch里面有64个样本，那么规范化输入的这64个样本各自经过这个神经元后的值（64维），LayerNorm则是针对单个样本，不依赖于其他数据，常被用于小mini-batch场景、动态网络场景和 RNN，特别是自然语言处理领域，就bert来说就是对每层输出的隐层向量（768维）做规范化，图像领域用BN比较多的原因是因为每一个卷积核的参数在不同位置的神经元当中是共享的，因此也应该被一起规范化。

### 23、self-attention相比lstm优点是什么？

bert通过使用self-attention + position embedding对序列进行编码，lstm的计算过程是从左到右从上到下（如果是多层lstm的话），后一个时间节点的emb需要等前面的算完，而bert这种方式相当于并行计算，虽然模型复杂了很多，速度其实差不多。

