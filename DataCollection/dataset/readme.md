1.数据下载：手写数字数据集原始地址：http://yann.lecun.com/exdb/mnist/

2.数据转换：下载后为.gz文件，解压后的文件为ubyte格式，在使用前需要进行转换提取成我们所需的图片格式，具体参考：https://blog.csdn.net/weixin_40522523/article/details/82823812

3.数据说明：
- 训练集图像：t10k-images.idx3-ubyte
- 训练集标签：t10k-labels.idx1-ubyte
- 测试集图像：train-images.idx3-ubyte
- 测试集标签：train-labels.idx1-ubyte

训练集 60000万张手写数字，测试集100000万张手写数字，从0到9十个类别