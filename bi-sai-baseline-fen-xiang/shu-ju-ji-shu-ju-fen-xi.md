# 数据集数据分析

数据分析采用splunk软件

**1.对话轮数分布**

source="data\_train.tsv" host="iZ2ze5v7m532iod14ldw49Z" sourcetype="tsv" \| stats count\(text\) as turn by sid \| stats count\(sid\)  by turn \| sort +turn

![&#x5BF9;&#x8BDD;&#x8F6E;&#x6570;-&#x5BF9;&#x8BDD;session&#x6570;](../.gitbook/assets/image%20%289%29.png)

可以知道大部分数据的对话session为30轮以下。

**2.**   **句子长短分布：**

source="data\_train.tsv" host="iZ2ze5v7m532iod14ldw49Z" sourcetype="tsv" \| eval length =  len\(text\) \| stats count\(text\)  by length \| sort +length

![&#x53E5;&#x5B50;&#x957F;&#x5EA6;-&#x53E5;&#x5B50;&#x6570;](../.gitbook/assets/image%20%2810%29.png)

**3.**   **图片问句：**

source="data\_train.tsv" host="iZ2ze5v7m532iod14ldw49Z" sourcetype="tsv" \| search text = "\*.jpg"

总数：379,020 

占比： 379,020 / 4,113,458 = **9.2%**

**4.**   **特殊字段：**

表情： \#E-s\[数字x\] \#E-2\[数字x\] 等一系列数字

标点符号

停用词

脱敏信息：

![&#x8131;&#x654F;&#x4FE1;&#x606F;](../.gitbook/assets/image%20%288%29.png)

无脱敏表情：

![&#x65E0;&#x8131;&#x654F;&#x8868;&#x60C5;](../.gitbook/assets/image%20%287%29.png)

url链接：&lt;url&gt;



