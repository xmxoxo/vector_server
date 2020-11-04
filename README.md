# 通用向量搜索服务


**通用向量搜索服务 VectorServer**

基于faiss搭建的通用向量搜索服务，服务加载向量持久化文件, 同时可指定加载数据文件；
通过faiss索引到内存，再通过flask提供API通用接口。

- API接口提供：
	* 按向量搜索向量；
	* 按索引号搜索向量；
	* 计算两个向量的余弦距离；
- 如果指定了数据文件，接口同时还会返回向量索引号对应的数据内容；
- 支持GPU

向量搜索服务应用广泛，可以应用在词库搜索，文档搜索，智能机器人等各种场景。

## 文件说明

| 文件名 | 文件说明 |
|--|--|
| VecSearch_faiss_server.py |服务主程序 |
|test.py | 测试程序 |
| /images | 截图目录 |


## 使用说明

向量搜索服务启动时需要两个参数：

1. 向量文件。格式为.npy文件，即把向量直接用`np.save(filename,v)`，保存的文件；
2. 数据文件。数据文件为可选，如果没有指定数据文件，则接口只会返回索引号；
当指定了数据文件后，接口会同时使用txt字段返回对应的文本数据；
数据文件格式为纯文本格式，内容按行分隔，与向量文件的顺序对应，
例如对应的句子，词语，或者其它的业务信息。



### 服务程序


服务端主程序使用说明
```
usage: VecSearch_faiss_server.py [-h] [-npy NPY] [-datfile DATFILE]
                                 [-port PORT] [-metric {L2,INNER_PRODUCT}]
                                 [-debug DEBUG] [--gpu GPU]

faiss向量搜索服务 V-1.0.1

optional arguments:
  -h, --help            show this help message and exit
  -npy NPY              向量数据文件名,默认vector.npy
  -datfile DATFILE      文本数据文件名
  -port PORT            监听端口，默认7800
  -metric {L2,INNER_PRODUCT}
                        计算方法:L2=欧式距离；INNER_PRODUCT=向量内积(默认)
  -debug DEBUG          是否调试模式，默认=1
  --gpu GPU             使用GPU,-1=不使用（默认），0=使用第1个，>0=使用全部

```

**注明**
当使用内积的方式时，服务会自动对向量进行归一化处理，
这样输出的结果可以认为就是余弦相似度；

### 测试程序

```
usage: test.py [-h] [-test_total TEST_TOTAL] [-test_dim TEST_DIM]
               [-test_times TEST_TIMES] [-test_topn TEST_TOPN]

faiss向量搜索服务端测试 V-1.0.2

optional arguments:
  -h, --help            show this help message and exit
  -test_total TEST_TOTAL
                        测试数据条数，默认10万
  -test_dim TEST_DIM    测试数据维度，默认768
  -test_times TEST_TIMES
                        测试次数，默认1000
  -test_topn TEST_TOPN  测试返回条数，默认5

```

运行测试程序后，会自动生成向量文件，保存为`test_dat.npy`, 
默认参数下 10万条768维度的向量文件大小约为585MB

```
python test.py
```

## 使用案例: 腾讯词向量搜索服务

源码地址： https://github.com/xmxoxo/Tencent_ChineseEmbedding_Process/



