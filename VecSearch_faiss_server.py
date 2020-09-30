#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
# VecSearch faiss server
# pip install datasketch
# document, key:integer, vector:np
update: 2020/7/8 
'''

import argparse
import numpy as np
import time
import faiss 
import json
import logging
import os
import sys
import psutil
import traceback
import pandas as pd

from flask import Flask, request, render_template, jsonify, abort
from flask import url_for, Response, json

# 版本号
gblVersion = '1.0.2'

#-----------------------------------------
'''
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
# 余弦相似度
def CosSim(a, b):
    return 1-cosine(a, b)

def CosSim_sk(a,b):
    score = cosine_similarity([a,b])[0,1]
    return score
'''

def CosSim_dot(a, b):
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return score
# -----------------------------------------

# 读入文件
def readtxtfile(fname,encoding='utf-8'):
    pass
    try:
        with open(fname,'r',encoding=encoding) as f:  
            data=f.read()
        return data
    except :
        return ''

'''
# 加载数据文件,  读取待处理的文本数据
数据格式：有标题行，TAB分隔
参数： 
    dat_file    数据文件
    column      提取第几列，从0开始，默认=0
    header      是否有表头，默认有=0
'''
def load_data (dat_file, column=0, header=0):
    train = pd.read_csv(dat_file, sep='\t', header=header)
    print('数据大小:', train.shape)
    # 会有空数据，处理成空格
    train.fillna(' ',inplace=True)
    questions = list(train[train.columns[column]])
    return questions

def MemoryUsed ():
    # 查看当前进程使用的内存情况
    import os, psutil
    process = psutil.Process(os.getpid())
    #info = 'Used Memory:',process.memory_info().rss / 1024 / 1024,'MB'
    info = 'Used Memory: %.3f MB' % (process.memory_info().rss / 1024 / 1024 )
    return info

# ----------------------------------------- 
'''
# 向量搜索类
'''
class VecSearch:
    def __init__(self, dim=10, nlist=100, metric='INNER_PRODUCT', gpu=-1): #normalize=True,
        self.dim = dim                          # 向量维度
        self.nlist = nlist                      # 聚类中心的个数
        #self.index = faiss.IndexFlatL2(dim)    # build the index
        quantizer = faiss.IndexFlatL2(dim)      # the other index

        # faiss.METRIC_L2: faiss定义了两种衡量相似度的方法(metric)，
        # 分别为faiss.METRIC_L2 欧式距离、 faiss.METRIC_INNER_PRODUCT 向量内积
        # here we specify METRIC_L2, by default it performs inner-product search
        
        # 创建时可指定计算方法，默认使用 faiss.METRIC_L2 欧式距离
        met = faiss.METRIC_INNER_PRODUCT
        if metric=='L2':
            met = faiss.METRIC_L2
            normalize = False
        if metric=='INNER_PRODUCT': # 内积时自动归一化
            met = faiss.METRIC_INNER_PRODUCT
            normalize = True
        
        # 是否归一化处理
        self.normalize = normalize
        
        # 生成索引
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, met)

        # 增加GPU迁移处理 2020/9/24
        try:
            if gpu>=0:
                if gpu==0:
                    # use a single GPU
                    res = faiss.StandardGpuResources()  
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                else:
                    gpu_index = faiss.index_cpu_to_all_gpus(self.index)
            
                self.index = gpu_index
        except :
            pass

        # data 
        self.xb = None
    
    # 返回当前总共有多少个值
    def curr_items(self):
        return self.xb.shape[0]

    # 清空数据
    def reset (self):
        pass
        self.xb = None

    # 添加向量，可批量添加，编号是按添加的顺序；
    # 参数: vector, 大小是(N, dim)
    # 返回结果：索引号区间, 例如 (0,8), (20,100)
    def add (self, vector):
        if not vector.dtype == 'float32':
            vector = vector.astype('float32')
        
        # 归一化处理 2020/9/7
        if self.normalize:
            faiss.normalize_L2(vector)

        if self.xb is None:
            prepos = 0
            # vector = vector[np.newaxis, :]   
            self.xb = vector.copy()
        else:
            prepos = self.xb.shape[0]
            self.xb = np.vstack((self.xb,q))
        
        return (prepos, self.xb.shape[0]-1)

    # 添加后开始训练
    def reindex(self):
        self.index.train(self.xb)
        self.index.add(self.xb)  # add may be a bit slower as well
    
    '''
    # 查找向量, 可以批量查找，
    # 参数：
        query: 搜索向量，大小=(N,dim)
        top:  返回多少个
        nprobe： 聚类中心个数,默认=1
        ret_vec: 是否返回向量结果, 默认=0否
        index:   可按索引号搜索向量，没传时使用query 
    # 返回： 
        距离D, 索引号I, 向量V  格式为(np.array, np.array, np.array)
    '''
    def search(self, query, top=5, nprobe=1, ret_vec=0, index=None): #
        D, I, V = [],[],[]

        # 查找聚类中心的个数，默认为1个。
        self.index.nprobe = nprobe #self.nlist 
        '''
        '''
        # 如果指定了索引号，则使用索引号指定的向量 2020/9/10
        if index:
            query = self.xb[index,:]
            # 如果是单条查询，把向量处理成二维 
            if len(query.shape)==1:
                query = query[np.newaxis, :]
        else:
            # 如果是单条查询，把向量处理成二维 
            if len(query.shape)==1:
                query = query[np.newaxis, :]
            if not query.dtype == 'float32':
                query = query.astype('float32')
            #print(query.shape)
            
            # 向量归一化
            if self.normalize:
                faiss.normalize_L2(query)
        
        # print('q,n:', (query, top) )
        # 查询
        D, I = self.index.search(query, top)
        # 添加向量输出 2020/9/7
        V = []
        if ret_vec:
            V = self.xb[I,:]

        return D, I, V

    '''
    按索引号返回向量, 2020/9/8 add by xmxoxo
    '''
    def get_vector (self, index):
        v = None
        #if 0<=index<=self.xb.shape[0]:
        try:
            v = self.xb[index,:]
        except :
            v = None
        return v

# -----------------------------------------
# 字符串反序列化为np.array
def str2array (txt):
    ret = []
    try:
        ret = json.loads(txt)
    except :
        pass
        print('Error in str2array')
    return np.array(ret)

# -----------------------------------------
# Flask 服务端
def HttpServer (args):
    # 参数处理
    npy = args.npy
    port = args.port
    debug = args.debug
    metric = args.metric
    datfile = args.datfile
    gpu = args.gpu

    logging.info( ('faiss向量搜索服务 v' + gblVersion ).center(40, '-') )

    # 加载数据文件 2020/9/10
    sentences = []
    if args.datfile:
        logging.info('正在加载数据文件...')
        sentences = readtxtfile(datfile).splitlines()
        #sentences = load_data(datfile)
        if not sentences:
            logging.info('加载数据失败,请检查数据文件...')
            #sys.exit()
        else:
            logging.info('加载数据文件完成: %s' % datfile )
            
    # 加载模型
    app = Flask(__name__)
    logging.info('正在加载向量并创建索引器...')
    start = time.time()
    vectors = np.load(npy)
    dim = vectors.shape[1]

    logging.info('向量文件:%s' % npy)
    logging.info('数据量:%d, 向量维度:%d' % vectors.shape)
    logging.info('距离计算方法:%s' % metric)
    logging.info('监听端口:%d' % port)
    

    # 创建搜索器
    vs = VecSearch(dim=dim, metric=metric, gpu=gpu)
    # 把向量添加到搜索器
    ret = vs.add(vectors)
    logging.debug(ret)
    # 添加数据后一定要索引
    vs.reindex()

    # 计算时间
    end = time.time()
    total_time = end - start
    logging.info('索引创建完成，用时:%4f秒' % (total_time) )

    # 显示内存使用
    logging.info(MemoryUsed())    
    # -----------------------------------------
    # API接口 
    
    '''
    # 查询两个词，并计算余弦相似度
    参数：words  用"|"分隔的两个词
    返回：simcos 余弦相似度
    '''
    @app.route('/api/v0.1/sim', methods=['POST'])
    def sim ():
        pass
        start = time.time()
        res = {}
        cos = None
        try:
            '''
            word_a = request.values.get('word_a')
            word_b = request.values.get('word_b')
            lstWord = [word_a, word_b]
            '''
            words = request.values.get('words')
            lstWord = words.split('|')

            index = [sentences.index(x) for x in lstWord]
            vector = vs.get_vector(index)
            # 计算余弦值
            cos = CosSim_dot(vector[0], vector[1])
        except :
            # 如果不存在要查的词，则报错
            cos = None

        # print(vector)
        # print(words, lstWord, index,cos)

        if cos:
            res["words"] = words
            res["index"] = str(index)
            res["simcos"] = str(cos)
            res["result"] = 'OK'
        else:
            res["result"] = 'Error'

        logging.info('查询用时:%.2f 毫秒' % ((time.time() - start)*1000) )
        return jsonify(res)
    # -----------------------------------------
    '''
    接口：按索引号返回向量
    调用方式：POST
    调用地址：/api/v0.1/vector
    参数： 
        index 索引号
        txt   文本
    返回：
        vector 向量
    '''
    @app.route('/api/v0.1/vector', methods=['POST'])
    def vector ():
        start = time.time()
        res = {}
        vector = None
        index = request.values.get('index')
        try:
            index = int(index)
            vector = vs.get_vector(index)
        except :
            vector = None
        
        txt = request.values.get('index')
        try:
            index = sentences.index(txt)
            vector = vs.get_vector(index)
        except :
            pass 

        # 同时返回对应的文本内容
        if txt=='':
            txt = sentences[index]

        if vector is None:
            res["result"] = 'Error'            
        else:
            res["index"] = index
            res["text"] = txt
            res["vector"] = str(list(vector))
            res["result"] = 'OK'

        logging.info('查询用时:%.2f 毫秒' % ((time.time() - start)*1000) )
        return jsonify(res)
    
    # -----------------------------------------
    '''
    接口： 查询单个向量并返回N个最接近的向量索引号
    调用方式：POST
    调用地址：/api/v0.1/query
    参数： 
        v: 要查询的向量, 序列化字符串
        n: 返回个数
        i: 如果指定索引号，则先按索引号获得向量；
        s: 文本
    返回：
        values: [D,I] 其中D和I都是array，
        样例： "values": "[[1.5547459e-16, 0.0006377562], [1, 15773]]"
        txt: 数据文件中对应的文本；

    说明：
        优先顺序为：索引 i > 文本 s > 向量v
    '''
    @app.route('/api/v0.1/query', methods=['POST'])
    def query ():
        start = time.time()
        res = {}
        top_n = 5
        if request.method != 'POST':
            return
    
        topn = request.values.get('n')
        
        '''
        if topn:
            if topn.isnumeric():
               top_n = int(topn)
        '''
        # 处理top_n参数
        if topn:
            top_n = int(topn) if topn.isdigit() else 5

        vec = request.values.get('v')
        txt = request.values.get('s')
        
        q,index = None, None
        # 增加索引号查询 2020/9/10
        index = request.values.get('i')
        
        '''
        if index:
            index = int(index) if index.isdigit() else None
            #q = vs.get_vector(index)
        '''
        try:
            index = int(index)
        except :
            index = None

        # 处理参数逻辑: 如果没有指定index则处理vec
        Err = 0
        if index is None:
            if not txt is None:
                try:
                    index = sentences.index(txt)
                except :
                    Err = 1
            else:
                if not vec is None:
                    # 把字符串转成 np.array
                    q = str2array(vec)
                    # print('qury=', q)
                    # logging.info ('查询向量: \n%s' % q)
                    # 判断维度是否正确
                    if q.shape[0]!= dim:
                        logging.info ('Error Dimension: require=%d, post=%d' % (dim,q.shape[0]))
                        res["result"] = "Error Dimension"
                        return jsonify(res)
                else:
                    Err = 1
        if Err:
            res["result"] = "Error requests: can not find text"
            return jsonify(res)

        # 查询并返回结果, 结果需要使用json.loads()转为list
        # 返回格式：[D,I,V]
        # 样例： "values": "[[1.5547459e-16, 0.0006377562], [1, 15773]]"
        # 增加了向量结果，可以选择输出；2020/9/7
        D, I, V = vs.search(q, top=top_n, nprobe=top_n, index=index)
        #value = [list(D[0]), list(I[0])]
        value = [D.tolist(), I.tolist()]
        
        # 增加输出原始文本 2020/9/10
        T = []
        if sentences:
            T = [sentences[i] for i in list(I[0])]
            
        #print(value)
        res["values"] = str(value)
        res["txt"] = str(T)
        res["result"] = 'OK'
        logging.info('查询结果:%s' % res)
        logging.info('查询用时:%.2f 毫秒' % ((time.time() - start)*1000) )
        return jsonify(res)

    # 启动服务 
    from gevent import pywsgi
    logging.info('Running under Linux...')
    app.debug = bool(debug)
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    server.serve_forever()


if __name__ == '__main__':
    #################################################################################################
    # 指定日志
    logging.basicConfig(level = logging.DEBUG,
                format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= os.path.join('./', 'app.log'),
                filemode='a')
    #################################################################################################
    # 定义一个StreamHandler，将 INFO 级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    #formatter = logging.Formatter('[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    formatter = logging.Formatter('[%(asctime)s]%(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################

    parser = argparse.ArgumentParser(description= ('faiss向量搜索服务 V-%s' % gblVersion) )
    parser.add_argument('-npy', default='vector.npy', type=str, help='向量数据文件名,默认vector.npy')
    parser.add_argument('-datfile', default='', type=str, help='文本数据文件名')  #required=True, 
    parser.add_argument('-port', default=7800, type=int, help='监听端口，默认7800')
    parser.add_argument('-metric', default='INNER_PRODUCT', choices=['L2','INNER_PRODUCT'], type=str, 
                    help='计算方法:L2=欧式距离；INNER_PRODUCT=向量内积(默认)')
    parser.add_argument('-debug', default=0, type=int, help='是否调试模式，默认=0')
    parser.add_argument('--gpu', default=-1, type=int, 
        help='使用GPU,-1=不使用（默认），0=使用第1个，>0=使用全部')

    args = parser.parse_args()
    if not os.path.exists(args.npy):
        logging.info('向量文件未找到，请检查参数...')
    else:
        HttpServer(args)


