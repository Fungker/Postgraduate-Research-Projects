import networkx as nx
import random
import numpy as np
import igraph as ig
import copy
from operator import itemgetter
import pandas as pd
import xgboost as xgb
import csv
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
import random
import re
import os
import networkx as nx
import pickle
from collections import Counter
from comm_detect import  *
from Help_Function import  *

import node2vec
#from scipy.stats import nanmean
from gensim.models import Word2Vec

#计算最短路径
def shortest_path(G, source, target):
    """
    G:  network
    source:  edges[0] in network (src)
    target:  edges[1] in network (dst)
    
    """
    try:
        # if the edge don't have sp, it will raise error.For example,the network is not connected
        sp = nx.shortest_path_length(G, source, target) 
    except:
        sp = 0
    return sp

#CN算法
def CN(G,nodeij):
    """
    common_neighors(G, nodeij)   
    calculate the common neighors of two nodes  
    Parameters
    ----------  
    G:            - networkx graph 
    nodeij        - pairs of nodes for sum the common neighbors    
    """
    node_i=nodeij[0]
    node_j=nodeij[1]
    neigh_i=set(G.neighbors(node_i))
    neigh_j=set(G.neighbors(node_j))
    neigh_ij=neigh_i.intersection(neigh_j)
    num_cn=len(neigh_ij)       
    return num_cn  

#LP1算法
def LP1(G,nodeij,alpha =0.1):
    """
    LP1(G,nodeij,alpha)    
    calculate the weighted common neighors of two nodes 
    Parameters
    ----------  
    G:            - networkx graph 
    nodeij        - pairs of nodes for sum the common neighbors 
    alpha:        - the alpha parameter in xiaoke's paper
    """
    node_i = nodeij[0]
    node_j = nodeij[1]
    paths_ij = nx.all_simple_paths(G,source=node_i,target=node_j,cutoff=3) #d<3的所有路径
    lp1 = 0.0
    list_path = list(paths_ij)
    if len(list_path) > 0:
        for path in list_path:
            if len(path) == 3: # 一阶共同邻居CN
                lp1 = lp1 + 1
            elif len(path) == 4:
                lp1 = lp1+alpha
    return lp1  

#PA算法
def PA(G,nodeij):
    """
    PA(G,nodeij)    
    calculate the common neighors of two nodes 
    Parameters
    ----------  
    G:            - networkx graph 
    nodeij        - pairs of nodes for sum the common neighbors 
    """
    node_i=nodeij[0]
    node_j=nodeij[1]
    degree_i=G.degree(node_i)
    degree_j=G.degree(node_j)
    a = degree_i*degree_j 
    return a 

#函数ccn2=cn+两个节点邻居在同一社区的数量
def CCN2(G,nodeij,comm_list): 
    node_i=nodeij[0]
    node_j=nodeij[1]
    neigh_i=list(G.neighbors(node_i))
    neigh_j=list(G.neighbors(node_j))  
    num_ccn=0
    for nodei in neigh_i:
        for nodej in neigh_j:
            for community in comm_list:
                if (int(nodei) in community) and (int(nodej) in community):
                    num_ccn+=1
    neigh_ij=set(neigh_i)&set(neigh_j)
    num_ccn+=len(neigh_ij)
    return num_ccn
	
def link_prediction1(Gn, data_name):
    """
    d= all
    data_name : 你的文件夹名字
    """

    times = 1.0 #循环x次，对预测结果取平均值

    auc_cn = []
    auc_lp1 = []
    auc_pa = []
    auc_ccn2 = []

    pre_cn = []
    pre_lp1 = []
    pre_pa = []
    pre_ccn2 = []
    


    for i in range(int(times)):
        print ('%d time'%i)
        #################################test_list and no_list#########
        train_graph = copy.deepcopy(Gn)
        train_list = list(train_graph.edges()) #训练集(全是正样本)
        print('length of train_graph edges = ', len(train_list))

        test_list = []  #测试集正样本
        length = int(0.1*len(train_list))  # 训练集正样本 ：测试集正样本= 9:1
        print(length)
        copy_train_graph = copy.deepcopy(train_graph)
        copy_train_list = copy.deepcopy(train_list)
        
        chosed_edges = []
        
        while len(test_list) < length:
            linkij = random.choice(train_list)
            if linkij not in chosed_edges:
                copy_train_graph.remove_edge(linkij[0], linkij[1])
                chosed_edges.append(linkij)
                if nx.has_path(copy_train_graph, linkij[0], linkij[1]):  #删除该边后，这两个节点之间是否可达（防止生成孤立节点）
                    test_list.append(linkij)
                    train_list.remove(linkij) #构建测试集正样本的时候，构建一条测试集，从训练集中删除一条数据，避免重复
                    train_graph.remove_edge(linkij[0],linkij[1])
        print('length of train_graph edges after remove = ', len(train_list))
                    
        #社团划分
        #####################使用训练集进行社团划分##########################
        nx.write_edgelist(train_graph,  './Data/{}/ig_train_graph.txt'.format(data_name),data=False)
        ig_train_graph = ig.Graph.Read_Edgelist('./Data/{}/ig_train_graph.txt'.format(data_name), directed=False)
        comm_list = fastgreedy_comm(ig_train_graph)   #社团划分结果 comm_list
        #保存社团划分结果
        with open('./Data/{}/comm_list.pkl'.format(data_name), 'wb') as f:
            pickle.dump(comm_list, f)
        #读取社团划分结果
#         pk = open('./Data/comm_list.pkl', 'rb')
#         comm_list = pickle.load(pk)
#         pk.close()       
        
        #保存训练集正样本
        df_train_pos = pd.DataFrame(list(train_list))
        df_train_pos.columns = ['src','dst']
        df_train_pos['label'] = 1
        df_train_pos['edge'] = list(zip(df_train_pos['src'], df_train_pos['dst']))
        df_train_pos.to_csv('./Data/' + data_name + '/train_test/train_pos_set.csv', index=False, sep = ' ')
        print('size of train pos set = ', len(df_train_pos))

        #存储测试集正样本
        df_test_pos = pd.DataFrame(test_list)
        df_test_pos.columns = ['src','dst']
        df_test_pos['label'] = 1
        df_test_pos['edge'] = list(zip(df_test_pos['src'], df_test_pos['dst']))
        df_test_pos.to_csv('./Data/' + data_name + '/train_test/test_pos_set.csv', index=False,sep =' ')
        print('size of test_pos_set = ', len(df_test_pos))



        no_list = []        #要预测的 （测试集负样本） 
        while len(no_list) < length:
            index_1 = random.choice(Gn.nodes())
            index_2 = random.choice(Gn.nodes())
            try:
                dict1 = Gn[index_1][index_2]    #有连边的返回一个字典， 无连边会报keyerror           
            except:
                if index_1 != index_2:
                    no_list.append((min(index_1,index_2),max(index_1,index_2))) 

        #存储测试集负样本
        df_test_neg = pd.DataFrame(no_list)
        df_test_neg.columns = ['src','dst']
        df_test_neg['label'] = 0
        df_test_neg['edge'] = list(zip(df_test_neg['src'], df_test_neg['dst']))
        df_test_neg.to_csv('./Data/' + data_name + '/train_test/test_neg_set.csv', index=False, sep =' ')
        print('size of test_neg_set = ', len(no_list))                

        
        #测试集正负样本合并
        df_Test = df_test_pos.append(df_test_neg)
        df_Test['CN'] = df_Test['edge'].apply(lambda x: CN(train_graph, x))
        df_Test['PA'] = df_Test['edge'].apply(lambda x: PA(train_graph, x))
        df_Test['LP1'] = df_Test['edge'].apply(lambda x: LP1(train_graph, x))
        df_Test['CCN2'] = df_Test['edge'].apply(lambda x : CCN2(train_graph, x, comm_list))
        df_Test.to_csv('./Data/' + data_name + '/train_test/Test_set.csv',index=False)
        
        real_auc_cn = []
        false_auc_cn = []
        
        real_auc_lp1 =[]
        false_auc_lp1 = []    
        
        real_auc_pa =[]
        false_auc_pa = []
        
        real_auc_ccn2 = []
        false_auc_ccn2 = []

        real_pre_cn = []
        false_pre_cn = []
        
        real_pre_lp1 = []
        false_pre_lp1 = []
        
        real_pre_pa = []
        false_pre_pa = []
        
        real_pre_ccn2 = []
        false_pre_ccn2 = []
        
        

        for linkij in test_list:
            cn = CN(train_graph,linkij)
            lp1 = LP1(train_graph, linkij)
            pa = PA(train_graph, linkij)
            ccn2 = CCN2(train_graph, linkij, comm_list)
            #cn
            real_auc_cn.append(cn)
            real_pre_cn.append((linkij, cn))
            #lp1
            real_auc_lp1.append(lp1)
            real_pre_lp1.append((linkij, lp1))
            #pa
            real_auc_pa.append(pa)
            real_pre_pa.append((linkij, pa))
            #CCN2 
            real_auc_ccn2.append(ccn2)
            real_pre_ccn2.append((linkij, ccn2))

        for linkij in no_list:
            cn = CN(train_graph,linkij)
            lp1 = LP1(train_graph, linkij)
            pa = PA(train_graph, linkij)
            ccn2 = CCN2(train_graph, linkij, comm_list)
            #cn
            false_auc_cn.append(cn)
            false_pre_cn.append((linkij,cn))
            #lp1
            false_auc_lp1.append(lp1)
            false_pre_lp1.append((linkij, lp1))
            #pa
            false_auc_pa.append(pa)
            false_pre_pa.append((linkij, pa))
            #ccn2
            false_auc_ccn2.append(ccn2)
            false_pre_ccn2.append((linkij, ccn2))
            
        #cn    
        auc_cn.append(AUC(real_auc_cn, false_auc_cn, 'CN',test_list,data_name))   
        pre_cn.append(PRECISION(real_pre_cn , false_pre_cn))
        #lp1
        auc_lp1.append(AUC(real_auc_lp1, false_auc_lp1, 'LP1',test_list, data_name))
        pre_lp1.append(PRECISION(real_pre_lp1, false_pre_lp1))
        #pa
        auc_pa.append(AUC(real_auc_pa, false_auc_pa, 'PA', test_list, data_name))
        pre_pa.append(PRECISION(real_pre_pa, false_pre_pa))
        #ccn2
        auc_ccn2.append(AUC(real_auc_ccn2, false_auc_ccn2, 'CCN2', test_list, data_name))
        pre_ccn2.append(PRECISION(real_pre_ccn2, false_pre_ccn2))

    print('---------------------------------------------------------')    
    print('mean of auc_cn = ', np.mean(auc_cn))
    print('mean of pre_cn = ', np.mean(pre_cn))
    print('---------------------------------------------------------')
    print('mean of auc_lp1 = ',np.mean(auc_lp1))
    print('mean of pre_lp1 = ', np.mean(pre_lp1))
    print('---------------------------------------------------------')
    print('mean of auc_pa = ',np.mean(auc_pa))
    print('mean of pre_pa = ', np.mean(pre_pa))
    print('---------------------------------------------------------')
    print('mean of auc_ccn2 = ',np.mean(auc_ccn2))
    print('mean of pre_ccn2 = ', np.mean(pre_ccn2))
    return train_graph, test_list