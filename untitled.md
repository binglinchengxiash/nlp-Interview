---
description: 2020-09-22更新
---

# 中文关键词抽取

## 中文关键词抽取的三种方法

**TF-IDF、textrank、word2vec**

### TF-IDF

TFIDF全称叫做term frequency–inverse document frequency，翻译过来可以叫做文本频率与逆文档频率, TFIDF就是为了表征一个token（可以是一个字或一个词）的重要程度。

主要用于提取文本中的关键词。

![](file:///C:/Users/songhuan/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

TF-IDF是对文本所有候选关键词进行加权处理，根据权值对关键词进行排序。

优点:简单快速，结果比较符合实际情况。

缺点:单纯以"词频"衡量一个词的重要性，不够全面，有时重要的词可能出现次数并不多。而且，这种算法无法体现词的位置信息，出现位置靠前的词与出现位置靠后的词，都被视为重要性相同，这是不正确的。

算法的关键词抽取步骤：

（1） 对于给定的文本D进行分词、词性标注和去除停用词等数据预处理操作。采用结巴分词，保留'n','nz','v','vd','vn','l','a','d'这几个词性的词语，最终得到n个候选关键词，即D=\[t1,t2,…,tn\];

（2） 计算词语ti 在文本D中的词频TF=Count\(ti\)/Count\(D\_word\)；

（3） 计算词语ti 在整个语料的IDF=log \(Dn /\(Dt +1\)\)，Dt 为语料库中词语ti 出现的文档个数；

（4） 计算得到词语ti 的TF-IDF=TF\*IDF，并重复（2）—（4）得到所有候选关键词的TF-IDF数值；

（5） 对候选关键词计算结果进行倒序排列，得到排名前TopN个词汇作为文本关键词。

实现代码：

```text
# 计算TF值
def tf(number_counts):
    for m in number_counts:
        fenmu = 0
        for j in number_counts[m]:
            fenmu += number_counts[m][j]#文章中出现词的总数
        for k in number_counts[m]:
            number_counts[m][k] = number_counts[m][k] / fenmu
            print(number_counts[m][k])
    return number_counts

# 计算IDF的值
def idf(number_counts):
    idf = {"行政": 0, "规划": 0, "政策": 0}
    for l in idf:
        count = 0
        D = 0
        for m in number_counts:
            D += 1
            if l in number_counts[m].keys():
                count += 1
        idf[l] = math.log(D /count)
    return idf
    
# 计算TF-IDF的值
def TFIDF(tf, idf):
    for m in tf:
        for k in idf:
            if k in tf[m].keys():
                tf[m][k] = tf[m][k] * idf[k]
    return tf
```

原始数据：

{'1': {'行政': 2, '规划': 4, '政策': 1}, '2': {'规划': 2, '政策': 1}, '3': {'行政': 1, '规划': 1}}

结果数据：

{'1': {'行政': 0.11584717374518982, '规划': 0.0, '政策': 0.05792358687259491}, '2': {'规划': 0.0, '政策': 0.13515503603605478}, '3': {'行政': 0.2027325540540822, '规划': 0.0}}

### Textrank

TextRank算法是基于PageRank算法，PageRank算法的核心思想是，认为网页重要性由两部分组成：

① 如果一个网页被大量其他网页链接到说明这个网页比较重要，即被链接网页的数量；

② 如果一个网页被排名很高的网页链接说明这个网页比较重要，即被链接网页的权重。

![](file:///C:/Users/songhuan/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

N为页面总数 

Textrank的思想是把文本拆分成词汇作为网络节点，组成词汇网络图模型，将词语间的相似关系看成是一种推荐或投票关系，使其可以计算每一个词语的重要性。

基于TextRank的文本关键词抽取是利用局部词汇关系，即共现窗口，对候选关键词进行排序，该方法的步骤如下：

（1） 对文本D进行分词、词性标注和去除停用词等数据预处理操作。本分采用结巴分词，保留'n','nz','v','vd','vn','l','a','d'这几个词性的词语，最终得到n个候选关键词，即D=\[t1,t2,…,tn\] ；

（2） 构建候选关键词图G=\(V,E\)，其中V为节点集（由候选关键词组成），并采用共现关系构造任两点之间的边，两个节点之间仅当它们对应的词汇在长度为K的窗口中共现则存在边，K表示窗口大小即最多共现K个词汇；

（3） 根据  公式  迭代  计算  各节点的权重，直至收敛；

（4） 对节点权重进行倒序排列，得到排名前TopN个词汇作为文本关键词。

说明：Jieba库中包含jieba.analyse.textrank函数可直接实现TextRank算法，本文采用该函数进行实验。

代码实现：

![](file:///C:/Users/songhuan/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

### Word2vec

对于用词向量表示的文本词语，通过K-Means算法对文章中的词进行聚类，选择聚类中心作为文章的一个主要关键词，计算其他词与聚类中心的距离即相似度，选择topN个距离聚类中心最近的词作为文本关键词，而这个词间相似度可用Word2Vec生成的向量计算得到。

假设Dn为测试语料的大小，使用该方法进行文本关键词抽取的步骤如下所示：

（1） 对Wiki中文语料进行Word2vec模型训练，参考文章“利用Python实现wiki中文语料的word2vec模型构建”（ [http://www.jianshu.com/p/ec27062bd453](http://www.jianshu.com/p/ec27062bd453) ），得到词向量文件“wiki.zh.text.vector”；

（2） 对于给定的文本D进行分词、词性标注、去重和去除停用词等数据预处理操作。本分采用结巴分词，保留'n','nz','v','vd','vn','l','a','d'这几个词性的词语，最终得到n个候选关键词，即D=\[t1,t2,…,tn\] ；

（3） 遍历候选关键词，从词向量文件中抽取候选关键词的词向量表示，即WV=\[v1，v2，…，vm\]；

（4） 对候选关键词进行K-Means聚类，得到各个类别的聚类中心；

（5） 计算各类别下，组内词语与聚类中心的距离（欧几里得距离），按聚类大小进行升序排序；

（6） 对候选关键词计算结果得到排名前TopN个词汇作为文本关键词。

步骤（4）中需要人为给定聚类的个数，本文测试语料是新闻文本，因此只需聚为1类，各位可根据自己的数据情况进行调整；步骤（5）中计算各词语与聚类中心的距离，常见的方法有欧式距离和曼哈顿距离，本文采用的是欧式距离，计算公式如下：

![IMG\_256](file:///C:/Users/songhuan/AppData/Local/Temp/msohtmlclip1/01/clip_image008.gif)

Word2vec代码实现

```text
from gensim.models.word2vec import Word2Vec
import gensim

# 读取数据，用gensim中的word2vec训练词向量

file = open('sentence.txt')

sss = []

while True:

    ss = file.readline().replace('\n', '').strip()

    if ss == '':

        break

    s1 = ss.split(" ")

    sss.append(s1)

file.close()

model = Word2Vec(size=200, workers=5, sg=1)  # 生成词向量为200维，考虑上下5个单词共10个单词，采用sg=1的方法也就是skip-gram

model.build_vocab(sss)

model.train(sss, total_examples=model.corpus_count, epochs=model.iter)

model.save('w2v_model')  # 保存模型 

new_model = gensim.models.Word2Vec.load('w2v_model')  # 调用模型

sim_words = new_model.most_similar(positive=['女人'])

for word, similarity in sim_words:

    print(word, similarity)  # 输出’女人‘相近的词语和概率

print(model['女孩'])   
```

三种方法对比：

本文总结了三种常用的抽取文本关键词的方法：TF-IDF、TextRank和Word2Vec词向量聚类，因本文使用的测试语料较为特殊且数量较少，未做相应的结果分析，根据观察可以发现，得到的十个文本关键词都包含有文本的主旨信息，其中TF-IDF和TextRank方法的结果较好，Word2Vec词向量聚类方法的效果不佳，这与文献\[8\]中的结论是一致的。

文献\[8\]中提到，对单文档直接应用Word2Vec词向量聚类方法时，选择聚类中心作为文本的关键词本身就是不准确的，因此与其距离最近的N个词语也不一定是关键词，因此用这种方法得到的结果效果不佳；而TextRank方法是基于图模型的排序算法，在单文档关键词抽取方面有较为稳定的效果，因此较多的论文是在TextRank的方法上进行改进而提升关键词抽取的准确率。

另外，本文的实验目的主要在于讲解三种方法的思路和流程，实验过程中的某些细节仍然可以改进。例如Word2Vec模型训练的原始语料可加入相应的专业性文本语料；标题文本往往包含文档的重要信息，可对标题文本包含的词语给予一定的初始权重；测试数据集可采集多个分类的长文本，与之对应的聚类算法KMeans\(\)函数中的n\_clusters参数就应当设置成分类的个数；根据文档的分词结果，去除掉所有文档中都包含某一出现频次超过指定阈值的词语。

