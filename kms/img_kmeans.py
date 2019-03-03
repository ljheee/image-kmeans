import PIL.Image as image;
import numpy as np;
from sklearn.cluster import KMeans;

def loadData(filepath):
    f = open(filepath, "rb")
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
            f.close()
    return np.mat(data), m, n


imgData, row, col = loadData("java.jpg")
label = KMeans(n_clusters=3).fit_predict(imgData)
label = label.reshape([row, col])
pic_new = image.new("L", (row, col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
pic_new.save("result.jpg", "JPEG")

print("-----------------finished---------------");
