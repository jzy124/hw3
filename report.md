[TOC]

#  Project 3：直方图图像增强

姓名：江朝昀

班级：自动化少61

学号：2140506069

提交日期：2019.3.17

##  摘要

使用python及其第三方库Skimage、Matplotlib，搭配OpenCV，对10幅经变亮或者变暗处理的图像和它们的4个源图像进行直方图处理。直方图处理包括绘制直方图、直方图均衡、直方图配准、局部直方图增强和利用直方图对图像进行分割。

 ##  一. 绘制图像的直方图

###  基本方法

Matplotlib.pyplot库中的hist函数可以绘制直方图。hist函数常用参数有：

* arr：需要绘制直方图的一维数组
* bins：可以简单理解为灰度级
* density：是否将得到的直方图向量归一化
* facecolor：直方图颜色

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.bmp')

plt.figure("lena")
arr = img.flatten()
plt.hist(arr, bins=256, density=1, edgecolor='None', facecolor='black')

plt.show()
```

图像与它们的直方图如下：

### lena

![lena](https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/lena.png)

![lena1](https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/lena1.png)

![lena2](https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/lena2.png)

![lena4](https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/lena4.png)

###  elain

![elain](https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/elain.png)

![elain1](https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/elain1.png)

![elain2](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/elain2.png>)

![elain3](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/elain3.png>)

###  citywall

![citywall](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/citywall.png>)

![citywall1](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/citywall1.png>)

![citywall2](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/citywall2.png>)

###  woman

![woman](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/woman.png>)

![woman1](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/woman1.png>)

![woman2](<https://github.com/jzy124/hw3/raw/74cace197de0da7df692ea83b33384a1b88927c3/pictures/1/woman2.png>)

##  二. 直方图均衡

###  基本方法

OpenCV库中自带函数可以对待处理图像做直方图均衡处理

```python
img2 = cv2.equalizeHist(img1)
```

###  lena

![lena](<https://github.com/jzy124/hw3/raw/master/pictures/2/lena.png>)

![lena1](<https://github.com/jzy124/hw3/raw/master/pictures/2/lena1.png>)

![lena2](<https://github.com/jzy124/hw3/raw/master/pictures/2/lena2.png>)

![lena4](<https://github.com/jzy124/hw3/raw/master/pictures/2/lena4.png>)

###  elain

![elain](<https://github.com/jzy124/hw3/raw/master/pictures/2/elain.png>)

![elain1](<https://github.com/jzy124/hw3/raw/master/pictures/2/elain1.png>)

![elain2](<https://github.com/jzy124/hw3/raw/master/pictures/2/elain2.png>)

![elain3](<https://github.com/jzy124/hw3/raw/master/pictures/2/elain3.png>)

###  citywall

![citywall](<https://github.com/jzy124/hw3/raw/master/pictures/2/citywall.png>)

![citywall1](<https://github.com/jzy124/hw3/raw/master/pictures/2/citywall1.png>)

![citywall2](<https://github.com/jzy124/hw3/raw/master/pictures/2/citywall2.png>)

###  woman

![woman](<https://github.com/jzy124/hw3/raw/master/pictures/2/woman.png>)

![woman1](<https://github.com/jzy124/hw3/raw/master/pictures/2/woman1.png>)

![woman2](<https://github.com/jzy124/hw3/raw/master/pictures/2/woman2.png>)

###  结果分析

观察实验结果可以发现，大部分图像经过直方图处理之后，对比度得到了明显的改善，它们的直方图从原先覆盖较窄的灰度范围扩展到较广的灰度范围。但是，lena1.bmp、elain1.bmp、citywall1.bmp和woman1.bmp这四幅图像的处理效果并没有很理想，它们没有得到显著的改善。

##  三. 直方图匹配

###  基本方法

直方图匹配的算法在教材中已经给出，主要公式如下：
$$
s_k=T(r_k)=（L-1）\sum_{j=0}^{k}{p_r}(r_j)=\frac{(L-1)}{MN}\sum_{j=0}^{k}{n_j},\quad k=0,1,2…,L-1
$$

$$
G(z_q)=(L-1)\sum_{i=0}^{q}{p_z(z_i)}
$$

通过匹配Sk和G(Zp)的值，可以找到Sk到Zp的映射。

核心步骤就是将Sk映射到Zq，这一步用直方图中得到的数值代入公式很容易求得。但是我前面的步骤中实际没有求得具体的直方图数组，而是用函数根据图像矩阵直接绘制。于是我参考网上的资料写了一个求图像的直方图数组的函数——arraytohist。求得原始图像直方图数值和指定的规定划图像直方图的数值后，构建函数histmatch可以进行匹配。histmatch中的核心算法根据上面两个公式进行设计，内容如下：

```python
m = np.zeros(256)
for i in range(256):
    idx = 0
    mini = 10
    for j in g:
        if np.fabs(g[j] - s[i]) < mini:
            mini = np.fabs(g[j] - s[i])
            idx = int(j)
    m[i] = idx
match = m[array]
return match
```

首先将各组图按照它们的原始图像进行直方图匹配。

###  lena

![lena1](<https://github.com/jzy124/hw3/raw/master/pictures/3/lena1.png>)

![lena2](<https://github.com/jzy124/hw3/raw/master/pictures/3/lena2.png>)

![lena4](<https://github.com/jzy124/hw3/raw/master/pictures/3/lena4.png>)

###  elain

![elain1](<https://github.com/jzy124/hw3/raw/master/pictures/3/elain1.png>)

![elain2](<https://github.com/jzy124/hw3/raw/master/pictures/3/elain2.png>)

![elain3](<https://github.com/jzy124/hw3/raw/master/pictures/3/elain3.png>)

###  citywall

![citywall1](<https://github.com/jzy124/hw3/raw/master/pictures/3/citywall1.png>)

![citywall2](<https://github.com/jzy124/hw3/raw/master/pictures/3/citywall2.png>)

###  woman

![woman1](<https://github.com/jzy124/hw3/raw/master/pictures/3/woman1.png>)

![woman2](<https://github.com/jzy124/hw3/raw/master/pictures/3/woman2.png>)

###  其他

+ lena2按照citywall进行匹配

![lena2-citywall](<https://github.com/jzy124/hw3/raw/master/pictures/3/lena2-citywall.png>)

+ elain3按照woman进行匹配

![elain3-woman](<https://github.com/jzy124/hw3/raw/master/pictures/3/elain3-woman.png>)

##  四. 局部直方图增强

###  基本方法

使用直方图统计的方法进行局部增强。首先明确需要增强的区域：对于两幅图都尝试增强其暗色区域，尽可能保留明亮区域不变。判断一个区域在点（x,y）是暗还是亮的方法是把局部平均灰度与全局平均灰度进行比较。全局平均灰度与局部平均灰度的计算公式分别为：
$$
m=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}{f(x,y)} \quad 全局平均灰度
$$
如果用r表示灰度值，用$p(r_i)$表示灰度$r_i​$出现的概率，全局平均灰度的计算公式还可以写成：
$$
m=\sum_{i=0}^{L-1}{r_i}{p(r_i)}\quad 全局平均灰度
$$

$$
m_{s_{xy}}=\sum_{i=0}^{L-1}{r_i}{p_{s_{xy}}}(r_i) \quad 局部平均灰度
$$

这样我们就有了增强方案的基础：如果$m_{s_{xy}}\leq k_0m_G$，其中$k_0$是一个小于1.0的正常数，那么我们将把点（x,y）处的像素考虑为处理的候选点。

因为我们感兴趣的是增强对比度低的区域，所以还需要一种度量方法来确定一个区域的对比度是否可以作为增强的候选点。判断的标准就是标准差。全局标准差和局部标准差的计算公式如下：
$$
\sigma_{G}^2=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}[{f(x,y)}-m]^2\quad全局方差
$$

$$
\sigma_{s_{xy}}=\sum_{i=0}^{L-1}({r_i}-{m_{s_{xy}}})^2p_{s_{xy}}(r_i)
$$

如果$\sigma_{s_{xy}}\leq{k_2}{\sigma_G}​$，$k_2​$为正常数，则认为在（x,y）处的像素是增强候选点。如果要增强暗区域则$k_2​$大于1.0，否则小于1.0。

最后我们需要限制能够接受处理的最低对比度值，否则会出现试图增强标准差为0 的恒定区域。要求${k_1}{\sigma_G}\leq{\sigma_{s_{xy}}}$,${k_1}\leq{k_2}$。满足所有条件的位于点（x,y）的像素，可以简单地通过将像素值乘以一个指定常数$E$来处理，以便相对于图像的其他部分增大（或减小）灰度值。不满足条件的点则保持不变。

本题中局部区域大小为7*7，按照教材上推荐的参数取值为$k_0=0.4,k_1=0.02,k_2=0.4,E=4​$。

主要程序如下：

```python
w_g, h_g = arr_g.shape
arr_s = np.zeros((7, 7))
for i in range(4, w_g - 3):
    for j in range(4, h_g - 3):
        for i1 in range(i - 3, i + 3):
            for j1 in range(j - 3, j + 3):
                arr_s[i1-i+3][j1-j+3] = arr_g[i1][j1]
        m_s, x_s = m_x(arr_s)
        if (m_s <= (k0 * m_g)) and (x_s <= (k2 * x_g)) and ((k1 * x_g) <= x_s):
             arr_g[i][j] = arr_g[i][j] * E
```

其中$m_g,x_g$是全局灰度均值和标准差，$m_s,x_s$是局部灰度均值和标准差。

###  lena

第一次尝试的结果是这样的：

![lena-1](<https://github.com/jzy124/hw3/raw/master/pictures/4/lena-1.png>)

可以看到，选择的区域基本没有问题，但是出现了很多增强过强的点，于是我减小$E$值，当$E=2$时

![lena-2](<https://github.com/jzy124/hw3/raw/master/pictures/4/lena-2.png>)

效果依然不是很好，尤其是背景中还是有很明显的噪点。于是我检查程序，发现我直接把$\sigma^2$当成标准差来进行计算了。开方，再次分别以$E=4，E=2$处理：

![lena-3](<https://github.com/jzy124/hw3/raw/master/pictures/4/lena-3.png>)

![lena-4](<https://github.com/jzy124/hw3/raw/master/pictures/4/lena-4.png>)

可以看到效果好了很多。为了防止有些点增强过强，我参考网上的资料改变一下增强计算公式：

```python
arr_g[i][j] = m_s + (arr_g[i][j] - m_s) * 2
```

![lena-5](<https://github.com/jzy124/hw3/raw/master/pictures/4/lena-5.png>)

JONG-SEN LEE提出过一种改进的方法，如下公式：

```python
arr_g[i][j] = m_s + (arr_g[i][j] - m_s) * (m_g / x_s)
```

![lena-6](<https://github.com/jzy124/hw3/raw/master/pictures/4/lena-6.png>)

P.S. 这个程序运行非常慢，平均五分钟以上

###  elain

用参数$k_0=0.4,k_1=0.02,k_2=0.4,E=4$对源图像进行处理

![elainE4](<https://github.com/jzy124/hw3/raw/master/pictures/4/elainE4.png>)

除了微弱的亮度增强以外看不到有明显的改善。

也尝试做了亮部改善，但是尝试了很多次也没有找到合适的参数。

##  五. 利用直方图对图像进行分割

利用函数cv2.threshold可以对图像进行分割（二值化）：

```python
ret, th = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
```

其中第二个参数是根据图像的直方图的峰值确定的，elain的峰值是130左右，woman的峰值是147左右。

另外cv2.threshold函数还可以进行Otsu’s阈值分割：

```python
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
```

用这两种方法分别处理elain.bmp和woman.bmp。

+ 简单二值化

![elain130](<https://github.com/jzy124/hw3/raw/master/pictures/5/elain130-255.png>)

![woman147](<https://github.com/jzy124/hw3/raw/master/pictures/5/woman147-255.png>)

+ Otsu's

![elain-otsu](<https://github.com/jzy124/hw3/raw/master/pictures/5/elain-otsu.png>)

![woman-otsu](<https://github.com/jzy124/hw3/raw/master/pictures/5/woman-otsu.png>)

可以看到，根据直方图峰值进行的图像分割效果要比Ostu‘s的效果要好一些。

##  总结

整个过程中遇到的一些问题：

+ 绘制直方图时，由于最开始没有添加edgecolor，直方图中某些非常窄的区域就显示不出来，造成观察的错误。
+ elain3和citywall2的直方图总是不能归一化显示。
+ skimage.exposure也可以绘制直方图，但是没有归一化
+ 读入图像的时候，如果在imread函数里增加参数cv2.IMREAD_GRAYSCALE，再imshow的话显示图像会出错，但是如果不加，相关灰度级计算会出问题。

###  参考资料

1. 图像均衡化及直方图匹配的详细实现，python：https://www.jianshu.com/p/3f6abf3eeba2
2. 基于局部均方差的图像局部对比度增强算法：https://blog.csdn.net/lz0499/article/details/77148829