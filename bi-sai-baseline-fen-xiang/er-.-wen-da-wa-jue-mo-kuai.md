---
description: 问答挖掘模块
---

# 二.问答挖掘模块

**1. 模块介绍**

该模块处理大赛发布的文本训练集数据，按照原始数据中的自然顺序将用户的消息当做问题，将客服的消息当做当前问题的应答，并根据人为制定的规则和闲聊分类模型的结果，将含有充足业务信息的问题和应答作为文档知识库保存下来。

该模型从./data文件夹下读取此次比赛的训练集数据，并以此构建QA文本知识库，（注：当前示例业务逻辑依赖于闲聊分类模型，请先训练闲聊模型）。

**2. 本模块输入**

train.txt

![train.txt&#x6587;&#x4EF6;&#x5185;&#x5BB9;&#x548C;&#x683C;&#x5F0F;](../.gitbook/assets/image%20%2815%29.png)

**3.本模块输出**

img\_QA\_dbs.txt 和 QA\_dbs.txt

![img\_QA\_dbs.txt &#x6587;&#x4EF6;&#x5185;&#x5BB9;&#x548C;&#x683C;&#x5F0F;](../.gitbook/assets/image%20%2810%29.png)

![QA\_dbs.txt&#x6587;&#x4EF6;&#x5185;&#x5BB9;&#x548C;&#x683C;&#x5F0F;](../.gitbook/assets/image%20%2826%29.png)

**4.源码解析**

\(1\).入口

对train.txt和dev.txt预处理，生成train\_items和dev\_items。

![&#x5165;&#x53E3;&#x51FD;&#x6570;](../.gitbook/assets/image%20%2811%29.png)

（2\).读取训练集数据每一行，保存到datalist。

![&#x8BFB;&#x53D6;train.txt&#x548C;dev.txt&#x6587;&#x4EF6;&#x6BCF;&#x4E00;&#x884C;](../.gitbook/assets/image%20%2830%29.png)

（3\).将原始数据按照session和问题、回答类型，用'\|\|\|'连接不同回车发送的内容，判断当前和下一句问答类型是否相同，将相同的连接起来。Add用于找下一个不同类型的问答用于连接。将QAAA，AAAQ等形式，转化为QAQA形式。

![](../.gitbook/assets/image%20%2812%29.png)

![](../.gitbook/assets/image%20%2819%29.png)

（4\).遍历全部（session, Q:xxx） \(session, A:xxx\),构建训练输入文件，Q，A，Context，其中'@@@'间隔Context里面不同的Q或者A。

构建question

![](../.gitbook/assets/image%20%2831%29.png)

构建answer

![](../.gitbook/assets/image%20%2822%29.png)

构建context

![](../.gitbook/assets/image%20%2818%29.png)

（5\).建立qa文本知识库

分为两步，首先是数据清洗，然后丢弃无意义的问题和答案。

![](../.gitbook/assets/image%20%2829%29.png)

I.  数据清洗（清理问题和回答）：去除句子中包含用户发起人工服务，售后资讯组，人工服务，售前资讯组，我要转人工，未购买-》售前咨询组，已购买-》售后服务组，以及去除图片。

![](../.gitbook/assets/image%20%286%29.png)





II.去除无意义的问题

问题是为空，&lt;url&gt;,&lt;num&gt;,问题长度小于4，问题包含谢谢，问题包含购买前咨询，或者问题是闲聊的问题。

![](../.gitbook/assets/image%20%2813%29.png)

III.去除无意义的回答

去除回答长度小于4的，回答包含欢迎光临，欢迎小主光临，稍等，有什么可以帮您的回答，回答是,,&lt;\#E-s&gt;的问题。

![](../.gitbook/assets/image%20%2832%29.png)

（6\).建立qa图文知识库

首先进行数据清理，然后去除不是图片的问题qa对，回答是图片的qa对，去除无意义的回答，写入图文知识库。多个图片可能对应同一个回答。

![](../.gitbook/assets/image%20%288%29.png)

I. 数据清理（清理问题和回答）:问题或者回答中包含去除包含用户发起人工服务，售后资讯组，人工服务，售前资讯组，我要转人工，未购买-》售前咨询组，已购买-》售后服务组。图片保留下来。

![](../.gitbook/assets/image%20%2827%29.png)

II.去除无意义回答的qa对

![](../.gitbook/assets/image%20%2824%29.png)

