# 一、模型探索
![[../../attachment/Pasted image 20241229164743.png]]
## 参数
![[../../attachment/Pasted image 20241229164927.png]]
``` python
from sklearn.cluster import KMeans
# 伪造测试样本
from sklearn.datasets import make_blobs
# 参数解释
# n_samples:样本总量
# n_features:特征维度
# centers：样本类别
X,y= make_blobs(n_sample=500,nfeatures=2,centers=4,random_state=2020)
# X 试题，y 答案
import matplotlib.pyplot as plt
# 将不类别的特征使用不同的颜色表示
color=['red','pink','orange','gray']

```

详见![[../../attachment/11.聚类算法.ipynb]]
# 二、模型评估

* KMeans的目标是确保簇内差异小，簇外差异大，我们就可以通过衡量簇内差异来衡量聚类的效果。簇内平方和/整体平方和是用距离来衡量簇内差异的指标，因此我们是否可以使用簇内平方和/整体平方和来作为聚类的衡量指标呢？簇内平方和越小越好吗？

![[../../attachment/Pasted image 20241229170913.png]]

不能：
![[../../attachment/Pasted image 20241229171012.png]]

## 轮廓系数

![[../../attachment/Pasted image 20241229171126.png]]

a：簇内差异
b：总体差异

期望：a越小越好，b越大越好，且a远离b

![[../../attachment/Pasted image 20241229171349.png]]

![[../../attachment/Pasted image 20241229171503.png]]




