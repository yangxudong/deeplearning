文本要实现的深度学习模型是阿里巴巴的算法工程师18年刚发表的论文《Perceive Your Users in Depth: Learning Universal User Representations from Multiple E-commerce Tasks》中提出的DUPN模型，实现过程和原始论文有一些不同之处，本实现采用了标准的LSTM模型作为网络的一部分，没有使用论文中修改过的Property Gated LSTM,另外，本模型的目标是训练一个分享率预估（类似于点击率预估）模型，并未用到多任务训练模式。

代码基于Tensorflow高阶API Estimator构建, 可以大规模分布式部署。

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
