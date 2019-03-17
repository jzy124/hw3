### 源代码：

#### 1.

```python
import cv2
import matplotlib.pyplot as plt

plt.figure("woman")
img1 = cv2.imread("woman.bmp")
plt.subplot(211)
plt.title('woman.bmp')
plt.imshow(img1, plt.cm.gray)
plt.axis('off')

arr1 = img1.flatten()
plt.subplot(212)
plt.hist(arr1, bins=256, density='true', edgecolor='black', facecolor='black')
plt.xlim(-1, 256)

plt.show()

```

#### 2.

```python
import cv2
import matplotlib.pyplot as plt

plt.figure("woman2")
img = cv2.imread("woman2.bmp")
plt.subplot(221)
plt.title('Original:woman2.bmp')
plt.imshow(img, plt.cm.gray)
plt.axis('off')

arr1 = img.flatten()
plt.subplot(222)
plt.hist(arr1, bins=256, density='true', edgecolor='black', facecolor='black')
plt.xlim(-1, 257)

img1 = cv2.imread("woman2.bmp", 0)
img2 = cv2.equalizeHist(img1)
plt.subplot(223)
plt.title('Histogram equalization')
plt.imshow(img2, plt.cm.gray)
plt.axis('off')

arr2 = img2.flatten()
plt.subplot(224)
plt.hist(arr2, bins=256, density='true', edgecolor='black', facecolor='black')
plt.xlim(-1, 257)

plt.show()

```

#### 3.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def arraytohist(array, nums):
    w, h = array.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if hist.get(array[i][j]) is None:
                hist[array[i][j]] = 0
            hist[array[i][j]] += 1
    n = w * h
    for key in hist.keys():
        hist[key] = hist[key]/n
    return hist


def histmatch(array, h_o, h_d):
    t = 0
    s = h_o.copy()
    for i in range(256):
        t += h_o[i]
        s[i] = t

    t = 0
    g = h_d.copy()
    for i in range(256):
        t += h_d[i]
        g[i] = t

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


plt.figure("woman2")

img_o = cv2.imread("woman2.bmp", cv2.IMREAD_GRAYSCALE)
arr1 = img_o.flatten()
plt.subplot(231)
plt.title("Original")
img_o1 = cv2.imread("woman2.bmp")
plt.imshow(img_o1, plt.cm.gray);
plt.axis('off')

img_d = cv2.imread("woman.bmp", cv2.IMREAD_GRAYSCALE)
arr2 = img_d.flatten()
plt.subplot(232)
plt.title("Specification")
img_d1 = cv2.imread("woman.bmp")
plt.imshow(img_d1, plt.cm.gray);
plt.axis('off')

plt.subplot(234)
plt.hist(arr1, bins=256, density='true', edgecolor='black', facecolor='black')
plt.xlim(-1, 257)
plt.subplot(235)
plt.hist(arr2, bins=256, density='true', edgecolor='black', facecolor='black')
plt.xlim(-1, 257)

arr_o = np.array(img_o)
arr_d = np.array(img_d)
hist_o = arraytohist(arr_o, 256)
hist_d = arraytohist(arr_d, 256)
img = histmatch(arr_o, hist_o, hist_d)
plt.subplot(233)
plt.title("Matched")
plt.imshow(img, cmap='gray');
plt.axis('off')

arr3 = img.flatten()
plt.subplot(236)
plt.hist(arr3, bins=256, density='true', edgecolor='black', facecolor='black')
plt.xlim(-1, 257)

plt.show()

```

#### 4.

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def arraytohist(array, nums):
    w, h = array.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if hist.get(array[i][j]) is None:
                hist[array[i][j]] = 0
            hist[array[i][j]] += 1
    n = w * h
    for key in hist.keys():
        hist[key] = hist[key] / n
    return hist


def m_x(array):
    hist = arraytohist(array, 256)
    m = 0
    for i in range(0, 255):
        m += i * hist[i]

    w, h = array.shape
    x = 0
    for ii in range(w):
        for jj in range(h):
            x += (array[ii][jj] - m) * (array[ii][jj] - m)
    x = math.sqrt(x / (w * h))
    return m, x


img = cv2.imread("elain.bmp", cv2.IMREAD_GRAYSCALE)
arr_g = np.array(img)
m_g, x_g = m_x(arr_g)

w_g, h_g = arr_g.shape
arr_s = np.zeros((7, 7))
for i in range(4, w_g - 3):
    for j in range(4, h_g - 3):
        for i1 in range(i - 3, i + 3):
            for j1 in range(j - 3, j + 3):
                arr_s[i1 - i + 3][j1 - j + 3] = arr_g[i1][j1]
        m_s, x_s = m_x(arr_s)
        if (m_s <= (0.4 * m_g)) and (x_s <= (1.5 * x_g)) and ((0.02 * x_g) <= x_s):
            arr_g[i][j] = arr_g[i][j] * 4

plt.figure("elain")
img1 = cv2.imread("elain.bmp")
plt.subplot(121)
plt.title("k0=0.4 k1=0.02 k2=1.5 E=4")
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.imshow(arr_g, cmap='gray')
plt.axis('off')

plt.show()

```

#### 5.

```python
import cv2
import matplotlib.pyplot as plt

plt.figure("elain")

img = cv2.imread('elain.bmp', cv2.IMREAD_GRAYSCALE)

ret, th = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
# ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波

plt.subplot(121)
plt.imshow(img, plt.cm.gray)
plt.axis('off')

plt.subplot(122)
plt.imshow(th, plt.cm.gray)
plt.axis('off')

plt.show()

```
