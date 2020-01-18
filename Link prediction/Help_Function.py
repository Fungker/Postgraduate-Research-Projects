
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
import matplotlib.pyplot as plt
import seaborn as sns


#构建训练集负样本，并且正负样本进行合并
def construct_Train(Gn, train_pos_set, data_name,test_list):
    train_neg_list = []

    G1 = copy.deepcopy(Gn)
    G1.remove_edges_from(test_list)

    while len(train_neg_list) < len(train_pos_set):
        index_1 = random.choice(G1.nodes())
        index_2 = random.choice(G1.nodes())
        try:
            dict1 = Gl[index_1][index_2]    #有连边的返回一个字典， 无连边会报keyerror           
        except:
    #         print(len(train_neg_list))
            if (index_1 != index_2) and ((index_1, index_2) not in train_neg_list):
                train_neg_list.append((index_1, index_2))
    
    print('size of train neg examples = ', len(train_neg_list))
    print('size of train neg example after droping duplicated = ', len(set(train_neg_list)))
    #存储训练集负样本
    df_train_neg = pd.DataFrame(train_neg_list)
    df_train_neg.columns = ['src','dst']
    df_train_neg['label'] = 0
    df_train_neg['edge'] = list(zip(df_train_neg['src'], df_train_neg['dst']))
    df_train_neg.to_csv('./Data/' + data_name + '/train_test/train_neg_set.csv', index=False ,sep =' ')
    
    #训练集正负样本合并
    train_pos_set['edge'] = train_pos_set['edge'].map(eval)
    df_Train = train_pos_set.append(df_train_neg)
    df_Train.to_csv('./Data/' + data_name + '/train_test/Train_set.csv', index=False, sep=' ')
    return df_Train
	
#xgboost  
def xgboost(train, test, save_path, test_list,num_round, data_name, params, embedding_algorithm_name):
    """
    train: train_set
    test: test_set
    save_path : path to save predict result
    test_list : as a benchmark, get the predict righ/wrong edges
    num_round : iteration time
    d_str : folder name,it is used to distinguish
    """
    #划分验证集
    trainxy, val = train_test_split(train, test_size = 0.1, random_state = 1)
    test_IMSI = test[['src','dst','label','distance']]

    #划分特征和标签
    y = trainxy.label
    val_y = val.label
    # 使用 distance 作为特征
    feature_name_train = ['distance']
    feature_name_test =['distance']
    feature_name_val =['distance']

    tests = test[feature_name_test]
    trains = trainxy[feature_name_train]
    vals = val[feature_name_val]

    dtrain = xgb.DMatrix(trains, label = y)
    dval = xgb.DMatrix(vals, label = val_y)
    dtest = xgb.DMatrix(tests)



    watchlist = [(dtrain, 'train'), (dval, 'val')]
    num_round = num_round
    curve_result = dict()
    model = xgb.train(params, dtrain, num_round, watchlist, evals_result=curve_result, verbose_eval=50)

    #绘制训练曲线
    plt.plot(range(len(curve_result['train']['error'])),curve_result['train']['error'],label='train')
    plt.plot(range(len(curve_result['train']['error'])),curve_result['val']['error'],label='val')
    plt.xlabel('number of trees')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    #predict test set
    test_y = model.predict(dtest, ntree_limit = model.best_ntree_limit)
    test_result = pd.DataFrame(test_IMSI, columns=['src','dst','label','distance']) 
    test_result['edge'] = list(zip(test_result['src'], test_result['dst']))
    test_result['prob'] = test_y


    #计算AUC* PRECISION 
    #从 y_score中 筛选出正负样本
    pos_y_score = test_result[test_result['label'] == 1]
    neg_y_score = test_result[test_result['label'] == 0]
    for d in [pos_y_score, neg_y_score]:
        d['edge'] = list(zip(d['src'], d['dst']))
        d['tuple'] = list(zip(d['edge'], d['prob']))
    real_yscore_auc = list(pos_y_score[pos_y_score['label'] == 1]['prob'])
    false_yscore_auc = list(neg_y_score[neg_y_score['label'] == 0]['prob'])
    real_yscore_pre = list(pos_y_score[pos_y_score['label'] == 1]['tuple'])
    false_yscore_pre = list(neg_y_score[neg_y_score['label'] == 0]['tuple'])
    #AUC 和 PRE
    node2vec_auc = AUC(real_yscore_auc, false_yscore_auc, embedding_algorithm_name, test_list, data_name)
    node2vec_pre = PRECISION(real_yscore_pre, false_yscore_pre)
    print('node2vec_AUC = ', node2vec_auc)
    print('node2vec_PRECISION = ', node2vec_pre)

    #概率大于0.5 设置为1
    test_result['predict_label'] = test_result['prob'].apply(lambda x: np.where(x > 0.5, 1, 0))
    test_result.to_csv(save_path, index=False, encoding='utf-8')
    return test_result
	
	
	
	
# 该函数为构建网络函数
def read_graph(filename,weighted,directed):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_adjlist(filename, nodetype = int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_adjlist(filename, nodetype = int, create_using = nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G
################################

def compute_vectors(nx_G, data_name):
    nx_G = read_graph(filename='./Data/' + data_name + '/train_graph.txt', weighted = False, directed = False)
    

    q_list = [0.25, 0.5, 1.0, 2.0]
    q = 0.25
    G = node2vec.Graph(nx_G, is_directed=False, p=1.0, q=q)  #调用node2vec算法，并设置p和q两个参数
    G.preprocess_transition_probs()
    #
    walks = G.simulate_walks(num_walks=10, walk_length=40)#num_walks每个节点作为开始节点的次数,#walk_length每次游走生成的节点序列的长度
    embedding_walks = [list(map(str, walk)) for walk in walks]
    #调用word2vec，得出节点向量 
    model = Word2Vec(embedding_walks, size=128, window=10, min_count=0, workers=8, iter=1,sg=1,hs=1)
    txt_link = './Data/' + data_name + '/vec/{}_vec.txt'.format(data_name)    
    model.wv.save_word2vec_format(txt_link)
	
	
	
#定义评价指标AUC
def AUC(real_edges,false_edges, metrics,test_list,data_name):
    AUC_result = 0.0
    right_edges0 = []
    random_edges1 = []
    wrong_edges2 = []
    for i in range(len(real_edges)):
        if real_edges[i] > false_edges[i]:
            AUC_result = AUC_result + 1
            right_edges0.append(test_list[i])
        elif real_edges[i] == false_edges[i]:
            AUC_result = AUC_result + 0.5
            random_edges1.append(test_list[i])
        elif real_edges[i] < false_edges[i]:
            wrong_edges2.append( test_list[i] )#因为这里索引的顺序和 计算CN的顺序相同
            
    #保存预测对的边，和不对的边
    right_data = pd.DataFrame(right_edges0)
    try:
        right_data.columns = ['src', 'dst']
        right_data.to_csv('./Data/' + data_name + '/metrics/'+ metrics+'/'+metrics+'_right_edges.csv', index=False)
    except:
        print('right_data = ' + 'empty')        
    
    random_data = pd.DataFrame(random_edges1)
    try:
        #空的df会报错
        random_data.columns =['src', 'dst']
        random_data.to_csv('./Data/' + data_name + '/metrics/'+ metrics+'/'+metrics+'_random_edges.csv', index=False)
    except:
        print('random_data = ' + 'empty')

    
    wrong_data = pd.DataFrame(wrong_edges2)
    try:
        wrong_data.columns =['src', 'dst']
        wrong_data.to_csv('./Data/' + data_name + '/metrics/'+ metrics+'/'+metrics+'_wrong_edges.csv', index=False)
    except:
        print('wrong_data = ' + 'empty')

    print('length of right edge {} = '.format(metrics), len(right_data))
    print('length of random edge {}= '.format(metrics), len(random_data))
    print('length of  wrong edge {} = '.format(metrics), len(wrong_data))
    print('------------------------------------------------------------------------')
    return AUC_result / len(real_edges)

#定义评价指标PRECISION
def PRECISION(real_edges,false_edges):
    topL = []
    cn_real = sorted(real_edges, key = lambda x:x[1], reverse = True)##reverse=True 从大到小的顺序
    cn_false  =sorted(false_edges, key = lambda x:x[1], reverse = True)
    i = 0
    j = 0
    L = 100
    while len(topL) < L:
        if cn_real[i][1] > cn_false[j][1]:
            topL.append(cn_real[i])
            i = i + 1
        elif cn_real[i][1] < cn_false[j][1]:
            topL.append(cn_false[j])
            j =j + 1
        else:
            same=[cn_real[i], cn_false[j]]
            a=random.choice(same)
            topL.append(a)
            same.remove(a)
            topL.append(same)
            i = i + 1
            j = j + 1     
    m = 0.0

    for i in range(L):
        if topL[i] in cn_real[0:L]:
            m = m + 1           
    return m / L   
	
	
def compute_distance(Train_set, Test_set, data_name, vector_type):
    if vector_type == 'Node2vec':
        vectors = pd.read_csv('./Data/' + data_name + '/vec/{}_vec.txt'.format(data_name), header = None, sep = ' ')
        vectors.columns = ['node'] + ['vec'+ str(i) for i in range(1, 129)]
        # 按照数据的顺序，进行建立字典，或者你可以按照节点进行排序。
        dict_node2vec = {}
        for index, n in enumerate(list(vectors['node'])):
            dict_node2vec[n] = list(vectors.iloc[index, 1:129])
        ### 进行数字典的映射
        for data_set in [Train_set, Test_set]:
            data_set['src_vec'] = data_set['src'].map(dict_node2vec)
            data_set['dst_vec'] = data_set['dst'].map(dict_node2vec)
            data_set['src_dst_vec'] = list(zip(data_set['src_vec'], data_set['dst_vec']))  
            data_set['distance'] = data_set['src_dst_vec'].apply(lambda x: np.linalg.norm( np.array(x[0]) - np.array(x[1])))  
            del data_set['src_vec']
            del data_set['dst_vec']
            del data_set['src_dst_vec']
            
        Train_set.to_csv('./Data/' + data_name + '/train_test/Train_distance_Node2vec.csv', index=False)
        Test_set.to_csv('./Data/' + data_name + '/train_test/Test_distance_Node2vec.csv', index=False)
	################################################		
    if vector_type == 'LINE':
        vectors = pd.read_csv('./Data/' + data_name + '/vec/{}_line_vec.txt'.format(data_name), header = None, sep = ' ')
        vectors.columns = ['node'] + ['vec'+ str(i) for i in range(1, 102)]
        del vectors['vec101']
        # 按照数据的顺序，进行建立字典，或者你可以按照节点进行排序。
        dict_node2vec = {}
        for index, n in enumerate(list(vectors['node'])):
            dict_node2vec[n] = list(vectors.iloc[index, 1:101])
        ### 进行数字典的映射
        for data_set in [Train_set, Test_set]:
            data_set['src_vec'] = data_set['src'].map(dict_node2vec)
            data_set['dst_vec'] = data_set['dst'].map(dict_node2vec)
            data_set['src_dst_vec'] = list(zip(data_set['src_vec'], data_set['dst_vec']))  
            data_set['distance'] = data_set['src_dst_vec'].apply(lambda x: np.linalg.norm( np.array(x[0]) - np.array(x[1])))  
            del data_set['src_vec']
            del data_set['dst_vec']
            del data_set['src_dst_vec']
            
        Train_set.to_csv('./Data/' + data_name + '/train_test/Train_distance_LINE.csv', index=False)
        Test_set.to_csv('./Data/' + data_name + '/train_test/Test_distance_LINE.csv', index=False)
	##########################################################	
    if vector_type == 'LargeVis':
        vectors = pd.read_csv('./Data/' + data_name + '/vec/{}_vec2D.txt'.format(data_name), header = None, sep = ' ')
        vectors.columns = ['node'] + ['vec'+ str(i) for i in range(1, 3)]    
        # 按照数据的顺序，进行建立字典，或者你可以按照节点进行排序。
        dict_node2vec = {}
        for index, n in enumerate(list(vectors['node'])):
            dict_node2vec[n] = list(vectors.iloc[index, 1:3])
        ### 进行数字典的映射
        for data_set in [Train_set, Test_set]:
            data_set['src_vec'] = data_set['src'].map(dict_node2vec)
            data_set['dst_vec'] = data_set['dst'].map(dict_node2vec)
            data_set['src_dst_vec'] = list(zip(data_set['src_vec'], data_set['dst_vec']))  
            data_set['distance'] = data_set['src_dst_vec'].apply(lambda x: np.linalg.norm( np.array(x[0]) - np.array(x[1])))  
            del data_set['src_vec']
            del data_set['dst_vec']
            del data_set['src_dst_vec']
            
        Train_set.to_csv('./Data/' + data_name + '/train_test/Train_distance_LargeVis.csv', index=False)
        Test_set.to_csv('./Data/' + data_name + '/train_test/Test_distance_LargeVis.csv', index=False)
    ##################################################################       
    if vector_type == 'GraphWave':
        vectors = pd.read_csv('./Data/' + data_name + '/vec/{}_Gw_vec.txt'.format(data_name), header = None, sep = ' ')
        vectors.columns = ['node'] + ['vec'+ str(i) for i in range(1, 101)]    
        # 按照数据的顺序，进行建立字典，或者你可以按照节点进行排序。
        dict_node2vec = {}
        for index, n in enumerate(list(vectors['node'])):
            dict_node2vec[n] = list(vectors.iloc[index, 1:101])
        ### 进行数字典的映射
        for data_set in [Train_set, Test_set]:
            data_set['src_vec'] = data_set['src'].map(dict_node2vec)
            data_set['dst_vec'] = data_set['dst'].map(dict_node2vec)
            data_set['src_dst_vec'] = list(zip(data_set['src_vec'], data_set['dst_vec']))  
            data_set['distance'] = data_set['src_dst_vec'].apply(lambda x: np.linalg.norm( np.array(x[0]) - np.array(x[1])))  
            del data_set['src_vec']
            del data_set['dst_vec']
            del data_set['src_dst_vec']
            
        Train_set.to_csv('./Data/' + data_name + '/train_test/Train_distance_GW.csv', index=False)
        Test_set.to_csv('./Data/' + data_name + '/train_test/Test_distance_GW.csv', index=False)     

    return Train_set, Test_set