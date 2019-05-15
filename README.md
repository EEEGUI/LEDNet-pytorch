[TOC]
# LEDNET: A LIGHTWEIGHT ENCODER-DECODER NETWORK FOR REAL-TIME SEMANTIC SEGMENTATION
LEDNet是最近新出的一个实时语义分割的网络，旨在保持一定分割精度的情况下，实现实时语义分割。文章的实验结果表明，在单张1080ti上能达到71FPS,在CityScapes数据集(Fine加Coarse)上能达到70.6%的mIoU,不使用额外的Coarse数据mIoU为69.2%，略低于ICNet(69.5%),但ICNet只能达到32FPS。ICNet凭借其良好的精度和实时性，已经在不少工业场景落地使用了,而LEDNet相比于ICNet又在计算速度上有翻倍的提升，值得一看。  
论文:https://arxiv.org/pdf/1905.02423.pdf  
原作者代码:https://github.com/xiaoyufenfei/LEDNet  
作者虽然公布了仓库，但代码还没上传，耐不住性子的我，反手写了一个:https://github.com/EEEGUI/LEDNet-pytorch

以下介绍文章亮点
## 1. 网络总体结构
网络总体总体上是一个Encoder-Decoder的结构(废话！网络名字就是这么叫的！)。Encoder中使用了自己设计的SS-nbt单元用于提取特征，Decoder中采用了多尺度的卷积处理特征。
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_003.png)  
各层结构如下;
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_004.png)  


## 2. Lightweight
上一部分说了题目中的Encoder-Decoder，这部分说说题目中的Lightweight，为什么网络的参数少。
### 2.1 channel 数目少
Encoder中包含大量的conv运算。假设在一个(N, H, W)的特征图上执行一个常规卷积运算，kernel-size为(3,3),输出的channe数仍为N,则参数量为$N\*3\*3\*N$，和channel的平方成正比。ResNet在最后一个block之后channel已经达到了2048，而LEDNet仅为128，参数减少量可想而知。

### 2.2 SS-nbt
SS-nbt的结构如下图所示，借鉴了ResNet、ShufferNet等网络中基本模块的思想。  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_00100.png)  
以下几个结构进一步减少了网络的参数
- Channle　split, 将输入的tensor一分为二,在两个分支上分别计算,进一步减少了channel的数目
- 用3\*1和1\*3的卷积提到了3\*3的卷积

channel shuffer用来消除channel split的影响，保证不同channel之间特征的组合交流。

## 3.剩下几个疑问
- Encoder中几个Unit空洞卷积的dilated r是如何设计的，看起来无迹可寻
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_006.png)  

- Decoder中注意力机制是如何体现的，特别是池化的那一个分支。
- 我自己实现的代码中，参数量为2.3M,比文章中的0.9M多了两倍还多，检查了好几遍，没发现自己代码有问题，等官方开源之后再对比看看。
- 在自己破笔记本(显卡MX150)上测试，LEDNet并没有比ICNet快,显存的占用也比ICNet大。


