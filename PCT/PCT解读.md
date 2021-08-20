Transformer具有很好的顺序不变性，而且在二维视觉任务上表现出了很好的效果，理论上可以代替卷积操作，因此transformer能够很好的应用在点云数据上。  

# 一、Transformer解读  
原论文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)  

## 1.编码器和解码器  
本部分参考链接：图解Transformer  
[原文](http://jalammar.github.io/illustrated-transformer/)  
[翻译](https://blog.csdn.net/longxinchen_ml/article/details/86533005)  
Transformer由一组encoder和一组decoder组成，在原文中，encoder和decoder的个数都是6个。6这个数字没有什么特别之处。也可以尝试其他的数字。  
![image](https://user-images.githubusercontent.com/74122331/129995453-a8c3fa54-5df4-4498-af![image](https://n.sinaimg.cn/sinacn20116/96/w1080h616/20190108/5bcc-hrkkwef7014930.jpg8d-fbd80f21a2a9.png)  
编码器由N = 6个相同层的堆叠组成。每层有两个子层。第一种是多头自关注(multi-head self-attention)机制，第二种是简单的、位置完全连接的前馈网络。两个字层之间使用了残差连接和层归一化。每个子层的输出是LayerNorm(x + Sublayer(x))，其中Sublayer(x)是子层本身实现的函数。每个子层和嵌入层的输出维度dmodel都是512.  

## 2.Attention机制  
本部分参考链接：
[Attention机制简单总结 - 知乎](https://zhuanlan.zhihu.com/p/46313756)  
[nlp中的Attention注意力机制+Transformer详解 - 知乎](https://zhuanlan.zhihu.com/p/53682800)  
[“变形金刚”为何强大：从模型到代码全面解析Google Tensor2Tensor系统](https://cloud.tencent.com/developer/article/1153079)  

一些名词解释：
标记嵌入 (token embedding)：也称词嵌入（word embedding），作用应该是将人类的语言映射到几何空间中。one-hot 编码得到的向量是二进制的、稀疏的（绝大部分元素都是 0）、维度很高的（维度大小等于词表中的单词个数）;词嵌入是低维的浮点数向量（即密集向量，与稀疏向量相对）。  

Query, Key和Value  
这是self-attention机制的数学化表达中用到的几个概念。  
可以把attention机制看成用一个query来检索一个memory，这个memory是一个字典，有key-value对。计算query和某个特定的key的相关度，用来确定这个key所对应的value的权重。这里的query、key和value都是向量。  
简化的attention计算分为3步：（1）计算query和key的相关度。（2）根据相关度做softmax处理，生成权重。（3）对value根据权重进行加权平均。
下面的图和部分内容引自上面的[图解Transformer](https://blog.csdn.net/longxinchen_ml/article/details/86533005):  
每个词所对应的这三个向量是由词嵌入向量和三个权重矩阵WQ、WK、WV相乘来确定的。  
![image](https://n.sinaimg.cn/sinacn20116/96/w1080h616/20190108/5bcc-hrkkwef7014930.jpg)  
词向量X1和权重矩阵WQ相乘得到q1，key和value的生成同理。  
self-attention的计算步骤如下：  
1.计算相关度：query向量和key向量点乘，得到一个打分，这个打分说明了这个单词对某个特定的位置有多重要（在这里是说明每个单词对第一个位置的重要性）  
![image](https://n.sinaimg.cn/sinacn20116/669/w746h723/20190108/ad95-hrkkwef7015564.jpg)  
2.打分除以8（维度dk的平方根，这里dk是64.这是一个默认值，可以修改）。这里除以√dk是因为长向量点积会很大，这样softmax的梯度不明显。这里可以参考[](https://blog.csdn.net/qq_37430422/article/details/105042303)。  
3.对处理过的打分做softmax，得到的值表示该单词对当下位置（Thinking）的贡献，第一个词（thinking本身）的贡献为0.88，第二个词的贡献为0.12。  
4.每个词的value向量乘以softmax后的权重。这样可以强化和特定位置相关性大的词造成的影响，弱化和这个位置相关性小的词造成的影响。  
5.每个词处理过的value向量加权求和。
实际上，为了加快速度，用矩阵来处理这些运算。上面几部
