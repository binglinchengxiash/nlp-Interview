---
description: 应答分类模块
---

# 四.应答分类模块

**1.模块介绍**

该模型以预训练的bert-base-chinese为基础，在大赛数据上自动构建Q-A对训练数据，使用bert中Next Sentence Prediction \(NSP\)的任务对模型进行fine-tune。

该模型可以对检索模型的召回结果进行排序， 从而得到最终的用户问题应答。

该模块以问答挖掘模块产生的文本问答知识库为数据资源，其中以知识库中存在的QA对，Qn与An问答对作为正例，以Qn与Arandom问答对作为负例，构建用于该模型训练的数据，在生成了模型所需的训练数据后，即可进行该模型的训练。

**2.Bert-NSP模型训练流程**

（1）  NSP任务介绍

多句子分类：CLS+句子A+SEP+句子B，利用CLS分类

![nsp&#x4EFB;&#x52A1;&#x56FE;](../.gitbook/assets/image%20%2823%29.png)

（2） 工作流程

I.从语料中提取两个句子 A 与 B ，50% 的概率 B 是 A 的下一个句子，50% 的概率 B 是一个随机选取的句子，以此为标注训练分类器。

 II.将 A 与 B 打包成一个序列（sequence）： \[CLS\] A \[SEP\] B \[SEP\] 。

 III.生成区间标识（segment labels），标识序列中 A 与 B 的位置。 \[CLS\] A \[SEP\] 的区域设为 0，B \[SEP\] 的区域设为 1： 0, 0..., 0, 1..., 1。 

IV.将序列与区间标识输入到模型，取 \[CLS\] 的表征训练 NSP 分类器。

**3.核心代码介绍**

\(1\).数据预处理

本模块作用：将问答模块产生的文本知识库数据，转化为BertForSequenceClassification要求的输入，主要是构造分类数据的正例和负例，并且划分训练集，验证集和测试集。

**I**.问答挖掘模块产生的文本知识库数据：

QA数据以tab分割

![](../.gitbook/assets/image%20%2828%29.png)

II.最终生成json数据

总共三份文件

![](../.gitbook/assets/image%20%2821%29.png)

Json文件内容：label:0代表不相关，1代表相关。

![](../.gitbook/assets/image%20%2816%29.png)

III.核心代码介绍

a.正例数据构造，列表中是字典数据。

