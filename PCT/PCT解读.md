Transformer具有很好的顺序不变性，而且在二维视觉任务上表现出了很好的效果，理论上可以代替卷积操作，因此transformer能够很好的应用在点云数据上。  

# 一、Transformer解读  
原论文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)  

## 1.编码器和解码器  
本部分参考链接：图解Transformer  
[原文](http://jalammar.github.io/illustrated-transformer/)  
[翻译](https://blog.csdn.net/longxinchen_ml/article/details/86533005)  
Transformer由一组encoder和一组decoder组成，在原文中，encoder和decoder的个数都是6个。6这个数字没有什么特别之处。也可以尝试其他的数字。   
![image](https://user-images.githubusercontent.com/74122331/131953867-1190ae76-2645-4c70-81bf-be0c1eca40cf.png)  
编码器由N = 6个相同层的堆叠组成。每层有两个子层。第一种是多头自关注(multi-head self-attention)机制，第二种是简单的、位置完全连接的前馈网络。两个子层之间使用了残差连接和层归一化。每个子层的输出是LayerNorm(x + Sublayer(x))，其中Sublayer(x)是子层本身实现的函数。每个子层和嵌入层的输出维度dmodel都是512.  

## 2.Attention机制  
本部分参考链接：
[Attention机制简单总结 - 知乎](https://zhuanlan.zhihu.com/p/46313756)  
[nlp中的Attention注意力机制+Transformer详解 - 知乎](https://zhuanlan.zhihu.com/p/53682800)  
[“变形金刚”为何强大：从模型到代码全面解析Google Tensor2Tensor系统](https://cloud.tencent.com/developer/article/1153079)  
[【经典精读】Transformer模型深度解读](https://zhuanlan.zhihu.com/p/104393915)  

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
实际上，为了加快速度，用矩阵来处理这些运算。上面几步都可以用一个公式来处理。
![B9E74FB18EFE4D042F6C4B4DB9AA4D07](https://user-images.githubusercontent.com/74122331/130198831-e75da75c-87cb-4f56-b963-c1c2d428257d.jpg)  
即  
![image](https://www.zhihu.com/equation?tex=Attention%28Q%2C+K%2C+V%29+%3D+softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5C%5C)

## 3.multi-head attention机制  
![image](https://user-images.githubusercontent.com/74122331/130307249-28fb31f4-4338-4f84-b70e-5c08838c2b70.png)  
我们发现使用不同的学习线性投影分别将查询、键和值时间线性投影到dk、dk和dv维度是有益的，而不是使用dmodel维度的键、值和查询执行单个注意函数。在查询、键和值的每个投影版本上，我们并行执行注意函数，生成dv维输出值。  
在原文中，实际上被使用的是性能更好的Multi-Head Attention机制。  
在该机制中，将V、K、Q经过多组线性变换后分别通过Attention得到相应的输出，随后将所有输出拼接在一起。  
![image](https://user-images.githubusercontent.com/74122331/131957789-36f2f502-57f9-45d8-aafe-7cc3396f1a37.png)  


## 4.位置编码  
在此之前，这个模型还少理解单词顺序的方法。
**（这可能是后面说的顺序不变性的来源？）**  
由于位置编码并不是PCT所需要的内容，再次不作展开。  

## 5.整体架构  

# 二、PCT解读  
[原论文](arxiv.org/pdf/2012.09688.pdf)  

## 1.PCT论文中的一些要点摘录  
PCT的核心思想是利用Transformer固有的顺序不变性，避免了定义点云数据顺序的需要，通过注意机制进行特征学习。注意力权重的分布与部件语义高度相关，它不会随着空间距离而严重衰减。  
由于点云和语言是完全不同的两个内容，因此PCT需要对Transformer进行如下的适应性改进：  
1.基于坐标的输入嵌入模块。  
Transformer使用位置编码来表示单词的顺序。点云具有无序性，不需要位置编码，而是把Transformer的位置编码和输入嵌入合并到输入嵌入模块中。由于每个点都有惟一的坐标，因此这种模块可以生成有区分性的编码。  
2.优化后的offset-attention模块  
3.邻居(Neibor embedding)嵌入模块  
单词本身具有语义特性，但是一个特定坐标和语义内容的关系是很弱的。attention机制注重全局特征，但可能会忽略局部特征。而局部特征对点云学习是必不可少的。作者使用了邻居嵌入模块来解决这个问题。  

## 2.网络架构  
![image](https://user-images.githubusercontent.com/74122331/130339677-e2b8f194-a562-499b-b4b9-1ac50ad91f08.png)  
编码器主要包括一个输入嵌入模块和四个堆叠的注意模块。解码器主要包括多个线性层（全连接层）。每个模块上方的数字表示其输出通道数。MA-Pool是Max-Pool和Average-Pool的结合，LBR是Linear, BatchNorm和ReLU层的结合。LBRD是LBR加上一个Dropout层。  
（1）编码器  
编码器与Transformer的编码器具有几乎相同的设计理念，只是不包括位置嵌入。
输入点云维度是N×d，表示N个d维的点；通过输入嵌入模块(input embedding)学习嵌入后的特征表Fe，维度为N×de
得到的逐点(point-wise)特征Fo则由如下公式生成：  
![image](https://user-images.githubusercontent.com/74122331/130343100-a950a492-bedd-47ed-bb7a-8346739145e9.png)
其中，ATi代表第i个关注层，每个关注层具有与其输入相同的输出维度，Wo是线性层的权重矩阵。Fo维度为N×do  
为了提取一个有效的表示点云的全局特征向量，在Fo后面接MA-Pool.  
（2）分类问题  
全局特征Fg（这里Fg应该是经过MA-Pool之后的Fo）被送到分类解码器中，分类解码器包括两个LBRD，dropout概率为0.5，然后由一个线性层输出概率。  
（3）分割问题  
部件分割要给每个点预测一个零件标签。先把全局特征Fg和逐点特征Fo连起来。分割网络和分类网络解码器的结构差不多，除了少了一个LBRD。  

## 3.注意力机制  
（1）Naive PCT  
由于查询、键和值矩阵由共享的对应线性变换矩阵和输入特征Fin确定，它们都是顺序无关的。此外，softmax和加权和都是与置换无关的算子。因此，整个自注意过程是排列不变的，这使得它非常适合于由点云呈现的无序的、不规则的域。  
（2）offset-attention  
![image](https://user-images.githubusercontent.com/74122331/130344281-741021ac-a274-495e-a447-79750a712984.png)
开关显示了自我注意或偏移注意的替代方案:虚线表示自我注意分支。  
偏移注意层通过元素减法计算自注意特征和输入特征之间的偏移(差异)。  

## 4.增强局部特征表示的邻域嵌入（neighbor embedding）  
![image](https://user-images.githubusercontent.com/74122331/130345625-da6bfe04-ed38-490d-be7a-ba004a216f74.png)  
左边:邻居嵌入架构；中间:SG（抽样和分组，sampling and grouping）模块，有Nin个输入点、Din个输入通道、k个邻居、Nout个输出采样点和Dout个输出通道；右上角:采样示例(彩球代表采样点)；右下角:用k-NN邻居分组的例子；LBR上面的数字:输出通道数。SG上面的数字:采样点数及其输出通道数。  

点嵌入的PCT可以有效提取全局特征，但是它忽略了局部特征，而这对点云处理是必要的。邻居嵌入模块包括两个LBR层和两个SG(采样和分组)层。像在CNN中一样，我们使用两个级联的SG层来逐渐扩大特征聚集过程中的感受野。在点云采样期间，SG层使用欧几里德距离为通过k-NN搜索分组的每个点聚合来自本地邻居的特征。  
更具体地说，假设SG层以具有N个点和相应特征F的点云P为输入，输出具有Ns个点及其相应聚集特征Fs的采样点云Ps。首先，我们采用最远点采样(FPS)算法将P降采样到Ps。然后，对于每个采样点p∈Ps，让knn(p，P)是它在P中的k近邻。然后，我们计算输出特征Fs如下:  
![image](https://user-images.githubusercontent.com/74122331/130345926-d3e6fe19-d219-45dd-91be-4ea4c0c751fb.png)  
F(p)是点p的输入特征，Fs(p)是采样点p的输出特征，MP是max-pooling，RP(x，k)是将一个向量x重复k次形成矩阵的算子。  
对于点云分类，我们只需要为所有点预测一个全局类，因此点云的大小在两个SG层内分别减少到512和256点。对于点云分割或正态估计，我们需要确定点状部分标签或正态，因此上述过程仅用于局部特征提取，而不减少点云大小，这可以通过将每个阶段的输出设置为大小仍然为n来实现。  
