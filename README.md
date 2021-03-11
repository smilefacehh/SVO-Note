# SVO-Note
SVO源码注释，半直接法VO。

SVO（Semi-direct monocular Visual Odometry），半直接法视觉里程计，只有前端。它结合了特征点和直接法，取特征点周围块的像素点用直接法进行跟踪，而不是跟踪所有的像素点。对于前端，包括典型的frame-to-frame，frame-to-map跟踪。对于建图部分，维护逆深度的分布（均值、方差），每次新来一帧，根据极线约束寻找投影点，然后三角化，更新逆深度的均值与方差，直到方差较小，认为该点的逆深度值比较准确了，便不再更新。主要知识点包括：

- 直接法，最小化光度误差，雅可比的构造
- 深度滤波器，理论推导

代码流程图

![Image](https://github.com/smilefacehh/SVO-Note/blob/main/SVO%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

贴一下论文给出的整体框架

![Image](https://github.com/smilefacehh/SVO-Note/blob/main/svo.png)

深度滤波示意图

![Image](https://github.com/smilefacehh/SVO-Note/blob/main/depthfilter.png)
