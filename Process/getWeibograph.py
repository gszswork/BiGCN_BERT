# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import pickle
cwd=os.getcwd()

dirname = 'data/'
def save_obj(obj, name ):
    with open(dirname+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( dirname + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None
        self.txt = ""

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=10000:
            wordFreq.append(freq)
            wordIndex.append(index-1)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    rootindex = 0
    for i in tree:                  # for i in dict: i是得到的key
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        nodeC.txt = tree[j]['txt']
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 10000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    ## 3. convert tree to matrix and edgematrix
    matrix=np.zeros([len(index2node),len(index2node)])
    raw=[]
    col=[]
    x_word=[]
    x_index=[]
    texts = []
    edgematrix=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                raw.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
        texts.append(index2node[index_i+1].txt)
    edgematrix.append(raw)
    edgematrix.append(col)
    return x_word, x_index, edgematrix, rootfeat, rootindex, texts

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 10000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x


def main():
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    treePath = os.path.join(cwd, 'data/Weibo/weibotree.txt')
    print("reading Weibo tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC, Vec, Text = line.split('\t')[0], line.split('\t')[1], \
                                         int(line.split('\t')[2]), line.split('\t')[3], line.split("\t")[4]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec, 'txt': Text}
    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, "data/Weibo/weibo_id_label.txt")
    print("loading weibo label:")
    event,y= [],[]
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid,label = line.split(' ')[0], line.split(' ')[1]
        labelDic[eid] = int(label)
        y.append(labelDic[eid])
        event.append(eid)
        if labelDic[eid]==0:
            l1 += 1
        if labelDic[eid]==1:
            l2 += 1

    print(len(labelDic),len(event),len(y))
    print(l1, l2)


    #results = Parallel(n_jobs=1, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    result_dict = {}
    for eid in tqdm(event):
        result_dict[eid] = {}
        tree = treeDic[eid]
        x_word, x_index, edgematrix, rootfeat, rootindex, texts = constructMat(tree)
        # edgematrix

        sent_id = bert_tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt', max_length=512)
        result_dict[eid]['inputs_ids'] = sent_id['input_ids'][:,:512]
        result_dict[eid]['attn_mask'] = sent_id['attention_mask'][:,:512]
        result_dict[eid]['root'] = rootfeat
        result_dict[eid]['edgeindex'] = edgematrix
        result_dict[eid]['rootindex'] = [rootindex]
        result_dict[eid]['y'] = labelDic[eid]

    print('length of saving dictionary: ', len(result_dict))
    save_obj(result_dict, 'tree_dict')

    return

if __name__ == '__main__':
    main()
