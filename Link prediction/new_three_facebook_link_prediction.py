# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:05:33 2018

@author: LENOVO
"""

import networkx as nx
import random
import numpy as np
#from comm_detect import *
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib
import copy
import os


###############################################
def CN(G,nodeij):
    node_i=nodeij[0]
    node_j=nodeij[1]
    neigh_ij=set(G.neighbors(node_i))&set(G.neighbors(node_j))
    num_ccn=len(neigh_ij)
    return num_ccn
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
#定义评价指标AUC
def AUC(real_edges,false_edges):
    AUC_result=0.0
    for i in range(len(real_edges)):
        if real_edges[i]>false_edges[i]:
            AUC_result=AUC_result+1
        elif real_edges[i]==false_edges[i]:
            AUC_result=AUC_result+0.5            
    return AUC_result/len(real_edges)
###############################
def PRECISION(real_edges,false_edges,L):
    topL=[]
    cn_real=sorted(real_edges, key=lambda x:x[1],reverse=True)
    cn_false=sorted(false_edges, key=lambda x:x[1],reverse=True)
    i=0
    j=0
    while len(topL) < L:
        if cn_real[i][1]>cn_false[j][1]:
            topL.append(cn_real[i])
            i=i+1
        elif cn_real[i][1]< cn_false[j][1]:
            topL.append(cn_false[j])
            j=j+1
        else:
            same=[cn_real[i], cn_false[j]]
            a=random.choice(same)
            topL.append(a)
            same.remove(a)
            topL.append(same)
            i=i+1
            j=j+1     
    m=0.0
    for i in range(L):
        if topL[i] in cn_real[0:L]:
            m=m+1           
    return m/L   

#对训练集与测试集的划分
def original_test_no_list(G):
    train_graph=copy.deepcopy(G) #整体的数据集
    train_list=train_graph.edges()  #将整个数据集中的边copy到训练集
    test_list=[]                    #测试集
    length=int(0.1*len(train_list))  #对测试集取长度；取整个训练集的10%
    while len(test_list)<length:
        linkij=random.choice(train_list)
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
            test_list.append(linkij)    #将边加到测试集中
            train_list.remove(linkij)   #将边从训练集中移除
            train_graph.remove_edge(linkij[0],linkij[1]) #
  
#选择与正样本同样大小的不存在连边集合作为测试集负样本。      
    no_list=[]                           ##实际不存在的连边   
    while len(no_list)<length:        #length 测试集的长度
        index_1=random.choice(G.nodes())
        index_2=random.choice(G.nodes())
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((index_1,index_2)) 
    return test_list,no_list,train_graph

############################################
def inside_test_no_list(all_edges,inside_edges,G):
    length=int(0.1*len(all_edges))     
    test_list=random.sample(inside_edges,length) 
    train_graph=copy.deepcopy(G)
    for linkij in test_list:
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
            train_graph.remove_edge(linkij[0],linkij[1]) 
            
    no_list=[]    
    while len(no_list)<length:
        index_1=random.choice(G.nodes())
        index_2=random.choice(G.nodes())
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((index_1,index_2))      
    return test_list,no_list,train_graph
####################################
def outside_test_no_list(all_edges,outside_edges,G):
    length=int(0.003*len(all_edges)) 
    test_list=random.sample(outside_edges,length) 
    train_graph=copy.deepcopy(G)
    for linkij in test_list:
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
            train_graph.remove_edge(linkij[0],linkij[1]) 
            
    no_list=[]    
    while len(no_list)<length:
        index_1=random.choice(G.nodes())
        index_2=random.choice(G.nodes())
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((index_1,index_2))      
    return test_list,no_list,train_graph


Gi_karate=ig.Graph.Read_Edgelist("ht09.txt")  #基于这些连边使用igraph创建一个新网络
Gi_karate=Gi_karate.as_undirected()
#print Gi_karate
Gn_karate= nx.read_edgelist("ht09.txt")
Gn_karate = Gn_karate.to_undirected()

#################调用已知社团划分################################
#a = open('348circles.txt','r')
#comm_list=[]
#for b in a.readlines():    
#    c = b.split()[1:]
#    d = list(map(int,c))
#    comm_list.append(d)
#print (comm_list)
##########################################################
test_list,no_list,train_graph=original_test_no_list(Gn_karate)

#new_G=nx.Graph()
#new_G.add_edges_from(edge_list)
nx.write_edgelist(train_graph,'train_graph.txt',data=False)
Gi_train_graph = ig.Graph.Read_Edgelist('train_graph.txt')
Gi_train_graph=Gi_train_graph.as_undirected()
#####################使用训练集进行社团划分##########################
comm_list=fastgreedy_comm(Gi_train_graph)   #社团划分结果 comm_list
###############################################################

L=5
times=5.0

#原始网络所有连边 
all_edges=Gn_karate.edges() 

inside_edges=[]
for edge in all_edges:
    for community in comm_list:
        if (int(edge[0]) in community) and (int(edge[1]) in community):
            inside_edges.append(edge)
inside_edges=list(set(inside_edges))
outside_edges=[i for i in all_edges if i not in inside_edges]

#########################################
ycn_auc=[]
ycn_auc_std=[]
yccn2_auc=[]
yccn2_auc_std=[]

ycn_pre=[]
ycn_pre_std=[]
yccn2_pre=[]
yccn2_pre_std=[]


for s in ['inside','outside','all']:
    print (s)
    auc_result_cn=[]
    auc_result_ccn2=[]
    pre_result_cn=[]
    pre_result_ccn2=[]
    
    for i in range(int(times)):
        
        print ('%d times'%i)
        
        if s=='inside':
            test_list,no_list,train_graph=inside_test_no_list(all_edges,inside_edges,Gn_karate)
        elif s=='outside':
            test_list,no_list,train_graph=outside_test_no_list(all_edges,outside_edges,Gn_karate)
        elif s=='all':
            test_list,no_list,train_graph=original_test_no_list(Gn_karate)
        
        ###############计算auc和precision############
        auc_cn_real=[]
        auc_ccn2_real=[]
        auc_cn_false=[]
        auc_ccn2_false=[]
        pre_cn_real=[]
        pre_ccn2_real=[]
        pre_cn_false=[]
        pre_ccn2_false=[]
        
        for linkij in test_list:
            cn=CN(train_graph,linkij)
            ccn2=CCN2(train_graph,linkij,comm_list)
            auc_cn_real.append(cn)
            auc_ccn2_real.append(ccn2)
            pre_cn_real.append((linkij,cn))
            pre_ccn2_real.append((linkij,ccn2))
        for linkij in no_list:
            cn=CN(train_graph,linkij)
            ccn2=CCN2(train_graph,linkij,comm_list)
            auc_cn_false.append(cn)
            auc_ccn2_false.append(ccn2)
            pre_cn_false.append((linkij,cn))
            pre_ccn2_false.append((linkij,ccn2))
        #===================================================
        auc_result_cn.append(AUC(auc_cn_real,auc_cn_false))
        auc_result_ccn2.append(AUC(auc_ccn2_real,auc_ccn2_false))
        pre_result_cn.append(PRECISION(pre_cn_real,pre_cn_false,L))
        pre_result_ccn2.append(PRECISION(pre_ccn2_real,pre_ccn2_false,L))
#=============================================================
    ycn_auc.append(sum(auc_result_cn)/times)
    ycn_auc_std.append(np.std(auc_result_cn))
    yccn2_auc.append(sum(auc_result_ccn2)/times)
    yccn2_auc_std.append(np.std(auc_result_ccn2))
    
    ycn_pre.append(sum(pre_result_cn)/times)
    ycn_pre_std.append(np.std(pre_result_cn))
    yccn2_pre.append(sum(pre_result_ccn2)/times)
    yccn2_pre_std.append(np.std(pre_result_ccn2))
#########################################

#print(yccn2_auc)
#print(yccn2_pre)

plt.figure(1,figsize=(7,9))
ax1=plt.subplot(211)
ax2=plt.subplot(212)
plt.subplots_adjust(wspace=0.4)
plt.rcParams['font.size']=15

X=np.arange(3)+1
plt.sca(ax1)
plt.bar(X-0.1,ycn_auc,width = 0.2,facecolor = 'yellowgreen',edgecolor = 'white',
        align="center",label='CN',yerr=ycn_auc_std)
plt.bar(X+0.1,yccn2_auc,width = 0.2,facecolor = 'pink',edgecolor = 'white',
        align="center",label='CCN2',yerr=yccn2_auc_std)
plt.ylim(0.45,1.01)
plt.xticks(X,['inside_edges','outside_edges','all_edges'])
plt.yticks(np.arange(0.45,0.9,0.1))
plt.text(0.2,0.98,'(a)')
plt.xlim(0.5,3.5)
plt.ylabel('AUC')
plt.legend(loc='best')

plt.sca(ax2)
plt.bar(X-0.1,ycn_pre,width = 0.2,facecolor = 'yellowgreen',edgecolor = 'white',
        align="center",yerr=ycn_pre_std)
plt.bar(X+0.1,yccn2_pre,width = 0.2,facecolor = 'pink',edgecolor = 'white',
        align="center",yerr=yccn2_pre_std)

plt.ylim(0.1,1.05)
plt.xticks(X,['inside_edges','outside_edges','all_edges'])
plt.yticks(np.arange(0.1,1,0.2))
plt.text(0.2,1,'(b)')
plt.xlim(0.5,3.5)
plt.ylabel('Precision')
plt.legend(loc='best')
#plt.savefig('686网络预测结果.pdf',bbox_inches='tight')
