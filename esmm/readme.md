本文是“基于Tensorflow高阶API构建大规模分布式深度学习模型系列”的第五篇，旨在通过一个完整的案例巩固一下前面几篇文章中提到的各类高阶API的使用方法，同时演示一下用tensorflow高阶API构建一个比较复杂的分布式深度学习模型的完整过程。

文本要实现的深度学习模式是阿里巴巴的算法工程师18年刚发表的论文《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》中提出的ESMM模型，关于该模型的详细介绍可以参考我之前的一篇文章：《CVR预估的新思路：完整空间多任务模型》。

## 使用说明

使用前需要先构建好tfrecord格式的样本数据。针对特定的任务，可以修改parse tfrecord的方法。

## 推荐阅读

1. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 开篇](https://zhuanlan.zhihu.com/p/38470806)
2. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列：基于Dataset API处理Input pipeline](https://zhuanlan.zhihu.com/p/38421397)
3. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 自定义Estimator（以文本分类CNN模型为例）](https://zhuanlan.zhihu.com/p/41473323)
4. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列:特征工程 Feature Column](https://zhuanlan.zhihu.com/p/41663141)
5. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列:CVR预估案例之ESMM模型](https://zhuanlan.zhihu.com/p/42214716)

## 后记

- 欢迎关注我的知乎专栏：[算法工程师的自我修养](https://zhuanlan.zhihu.com/yangxudong)
- 欢迎收藏我的个人博客，会不定期更新：[https://yangxudong.github.io](https://yangxudong.github.io)，
或者国内镜像：[https://xudongyang.coding.me](https://xudongyang.coding.me)
