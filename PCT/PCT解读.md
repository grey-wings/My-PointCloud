Transformer具有很好的顺序不变性，而且在二维视觉任务上表现出了很好的效果，理论上可以代替卷积操作，因此transformer能够很好的应用在点云数据上。  

# 一、Transformer解读  
原论文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)  

## 1.编码器和解码器  
本部分参考链接：  
[原文](http://jalammar.github.io/illustrated-transformer/)  
[翻译](https://blog.csdn.net/longxinchen_ml/article/details/86533005)  
Transformer由一组encoder和一组decoder组成，在原文中，encoder和decoder的个数都是6个。6这个数字没有什么特别之处。也可以尝试其他的数字。  
![image](https://user-images.githubusercontent.com/74122331/129995453-a8c3fa54-5df4-4498-af8d-fbd80f21a2a9.png)
编码器由N = 6个相同层的堆叠组成。每层有两个子层。第一种是多头自关注(multi-head self-attention)机制，第二种是简单的、位置完全连接的前馈网络。两个字层之间使用了残差连接和层归一化。每个子层的输出是LayerNorm(x + Sublayer(x))，其中Sublayer(x)是子层本身实现的函数。每个子层和嵌入层的输出维度dmodel都是512.  

## 2.Attention机制  
