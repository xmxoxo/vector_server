#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


import argparse
from VecSearch_faiss_server import *

# 测试  total=100000, dim=768, test_times=1000, topn=5 
def test_task (args):
    total = args.test_total
    dim = args.test_dim
    test_times = args.test_times
    top_n = args.test_topn
    gpu = args.gpu

    print('向量搜索服务测试-[faiss版]'.center(40,'='))
    # 随机生成向量 1百万
    # total = 1000000
    # 向量的维度
    # dim = 1024 #768
    print('随机生成%d个向量，维度：%d' % (total, dim), flush=True)
    #rng = np.random.RandomState(0)
    #X = rng.random_sample((total, dim))
    X = np.random.random((total, dim))
    # 局部做特殊处理
    X[:, 0] += np.arange(total) / 1000.
    # 保存数据 2020/5/22
    np.save('./test_dat.npy',X)

    #print('前10个向量为：')
    #print(X[:10])
    print('正在创建搜索器...')
    start = time.time()
    # 创建搜索器
    vs = VecSearch(dim=dim, gpu=gpu)
    # 把向量添加到搜索器
    ret = vs.add(X)
    print(ret)
    # 添加数据后一定要索引
    vs.reindex()

    # 计算时间
    end = time.time()
    total_time = end - start
    print('创建用时:%4f秒' % total_time)
    # 显示内存使用
    print(MemoryUsed())    
    # 进行测试
    print('单条查询测试'.center(40,'-'))
    #test_times = 1000
    #top_n = 100
    #Q = rng.random_sample((test_times, dim))
    Q = np.random.random((test_times, dim))
    Q[:, 0] += np.arange(test_times) / 1000.
    
    q = Q[0]
    #print('query:' , q)
    start = time.time()
    D, I, V = vs.search(q, top=top_n, nprobe=10)
    
    # 显示详细结果
    def showdetail (X,q,D,I):
        print('显示查询结果，并验证余弦相似度...')
        #for i in range(len(I[0])):
        for i,v in enumerate(I[0]):
            c = CosSim_dot(Q[0], X[v])
            r = (v, D[0][i], c) # CosSim_dot(Q[0], X[v]),
            print('索引号:%5d, 距离:%f, 余弦相似度:%f' % r )
            #rv = X[v][:10]
            #print('\n查询结果(超长只显示前10维:%s' % rv)

    showdetail (X,q,D,I)
    end = time.time()
    total_time = (end - start)*1000
    print('总用时:%d毫秒' % (total_time) )


    print('批量查询测试'.center(40,'-'))
    start = time.time()
    print('正在批量测试%d次，每次返回Top %d，请稍候...' % (test_times,top_n) )
    for i in range(test_times):
        D,I,V = vs.search(Q[i])
    end = time.time()
    total_time = (end - start)*1000
    print('总用时:%d毫秒, 平均用时:%4f毫秒' % (total_time, total_time/test_times) )
    #return 
        
    # 人工测试
    while 1:
        print('-'*40)
        txt = input("回车开始测试(Q退出)：").strip()
        if txt.upper()=='Q': break
      
        # 随机生成一个向量
        print('随机生成一个查询向量...')
        #q = rng.random_sample(dim)
        Q = np.random.random(dim)
        print("query:%s..." % q[:5])

        # 查询
        start = time.time()
        D, I, V = vs.search(q, top=top_n)
        print('查询结果:...')
        print('相似度:%s \n索引号:%s' % (str(D),str(I)) )
        end = time.time()
        total_time = end - start
        print('用时:%4f 毫秒' % (total_time*1000) )



if __name__ == '__main__':
    pass
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

    parser = argparse.ArgumentParser(description= ('faiss向量搜索服务端测试 V-%s' % gblVersion) )
    parser.add_argument('-gpu', default=-1, type=int, 
        help='使用GPU,-1=不使用（默认），0=使用第1个，>0=使用全部')
    
    parser.add_argument('-test_total', default=100000, type=int, help='测试数据条数，默认10万')
    parser.add_argument('-test_dim', default=768, type=int, help='测试数据维度，默认768')
    parser.add_argument('-test_times', default=1000, type=int, help='测试次数，默认1000')
    parser.add_argument('-test_topn', default=5, type=int, help='测试返回条数，默认5')
    
    args = parser.parse_args()
    test_task(args)
