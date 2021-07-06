import sys, os, json
import torch, numpy
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch.utils.data.sampler import WeightedRandomSampler

sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy, pickle
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel
from model.Weibo.layers import GraphAttentionLayer
device = 'cuda'
dirname = 'data/'


def save_obj(obj, name):
    with open(dirname + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(dirname + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        #self.conv1 = GCNConv(in_feats, hid_feats)
        #self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        # def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        #print(in_feats, hid_feats, out_feats)
        self.att1 = GraphAttentionLayer(in_features=in_feats, out_features=hid_feats, dropout=0, alpha=0.1)
        self.att2 = GraphAttentionLayer(in_features=hid_feats+in_feats, out_features=out_feats, dropout=0, alpha=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # torch.sparse_coo_tensor(edge_index, value=1, (len(x), len(x))
        #print(x.shape, edge_index.shape)
        adj_mat = torch.sparse_coo_tensor(edge_index,
                                          torch.tensor([1 for i in range(edge_index.shape[1])]).to(device),
                                          (x.shape[0], x.shape[0])).to_dense()

        x1 = copy.copy(x.float())

        #x = self.conv1(x, edge_index)
        x = self.att1(x, adj_mat)
        #print(x.shape)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        #print('shape before conv2', x.shape)
        x = self.att2(x, adj_mat)
        #print('shape after conv2: ', x.shape)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        #self.conv1 = GCNConv(in_feats, hid_feats)
        #self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.att1 = GraphAttentionLayer(in_features=in_feats, out_features=hid_feats, dropout=0, alpha=0.1)
        self.att2 = GraphAttentionLayer(in_features=hid_feats+in_feats, out_features=out_feats, dropout=0, alpha=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        adj_mat = torch.sparse_coo_tensor(edge_index,
                                          torch.tensor([1 for i in range(edge_index.shape[1])]).to(device),
                                          (x.shape[0], x.shape[0])).to_dense()
        x1 = copy.copy(x.float())
        #x = self.conv1(x, edge_index)
        x = self.att1(x, adj_mat)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        x = self.att2(x, adj_mat)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 2)

        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")

    def forward(self, data):
        bert_output = self.bert_model(data.input_ids, data.attn_mask)
        cont_reps = bert_output.last_hidden_state

        # print(cont_reps.shape[1])
        data.x = cont_reps[:, cont_reps.shape[1] - 1]
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class MyGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, tddroprate=0, budroprate=0, data_dict=None):
        self.fold_x = list(filter(lambda id: id in data_dict, fold_x))  # comment this line if error
        self.treeDic = treeDic
        self.data_dict = data_dict
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = self.data_dict[id]
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]

        '''
        Return: 
        input_ids, attn_mask: BERT model的输入
        edge_index: Top-Down 格式下的邻接关系，shape(2,n)，第一行为邻接矩阵的行，第二行为邻接矩阵的列.
        BU_edge_index: Botton-Up 的邻接关系， 在没有dropout的前提下，edge_index 和 BU_edge_index代表的邻接矩阵互为对方的转置.
        y: label
        root: 根节点的ids
        rootindex: 好像没有用到

        '''
        return Data(input_ids=data['inputs_ids'],
                    attn_mask=data['attn_mask'],
                    edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']), x=data['inputs_ids'],
                    rootindex=torch.LongTensor([int(data['rootindex'][0])]))


def train():
    lr = 0.00005
    weight_decay = 1e-4
    patience = 5
    n_epochs = 200
    batchsize = 1
    tddroprate = 0.0
    budroprate = 0.0
    datasetname = "Weibo"
    # iterations = int(sys.argv[1])
    model = "BiGCN"
    device = th.device('cuda')
    test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

    x_train_eid = load_train(datasetname)  # train 和 test 都是保存的tree id
    x_test_eid = load_test(datasetname)
    treeDict = loadTree(datasetname)
    data_dict = load_obj('tree_dict')

    filtered_data_dict = {}
    for key in data_dict.keys():
        shape_of_input = data_dict[key]['inputs_ids'].shape
        if shape_of_input[0] * shape_of_input[1] < 8000:
            filtered_data_dict[key] = data_dict[key]

    print('length of filtered data(remove too large samples): ', len(filtered_data_dict))
    traindata_list = MyGraphDataset(x_train_eid, treeDict, tddroprate=tddroprate, budroprate=budroprate,
                                    data_dict=filtered_data_dict)
    testdata_list = MyGraphDataset(x_test_eid, treeDict, data_dict=filtered_data_dict)
    print(len(traindata_list), len(testdata_list))
    model = Net(768, 64, 64).to(device)
    print('puting model to: ', device)

    BU_params = list(map(id, model.BUrumorGCN.att1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.att2.parameters()))
    BERT_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params and id(p) not in BERT_params, model.parameters())
    optimizer = th.optim.Adam([
        {'params': base_params},
        {'params': model.bert_model.parameters(), 'lr': lr / 10},
        #{'params': model.BUrumorGCN.att1.parameters(), 'lr': lr / 5},
        #{'params': model.BUrumorGCN.att2.parameters(), 'lr': lr / 5}
    ], lr=lr, weight_decay=weight_decay)

    train_epoch = 50
    for epoch in range(train_epoch):
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=1)
        tra_loss_sum = 0
        tra_data_len = len(traindata_list)
        model.train()
        avg_loss, avg_acc = [], []
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data = Batch_data.to(device)
            out_labels = model(Batch_data)
            label_y = Batch_data.y
            loss = F.nll_loss(out_labels, label_y)

            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(label_y).sum().item()
            train_acc = correct / len(label_y)
            avg_acc.append(train_acc)
            postfix = "Epoch {:05d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch,
                                                                                       loss.item(),
                                                                                       train_acc)
            tqdm_train_loader.set_postfix_str(postfix)
            tra_loss_sum += float(loss.to('cpu'))
            th.cuda.empty_cache()
        print('training loss: ', tra_loss_sum / tra_data_len)
        # 验证训练结果
        val_loss_sum = th.tensor(0.0).to(device)
        model.eval()
        with th.no_grad():
            tqdm_test_loader = tqdm(test_loader)

            truth, prediction = [], []
            for dev_data in tqdm_test_loader:
                dev_data = dev_data.to(device)
                dev_out = model(dev_data)
                dev_y = dev_data.y
                val_loss = F.nll_loss(dev_out, dev_data.y)
                val_loss_sum += val_loss
                # print(val_loss)
                # temp_val_losses.append(val_loss.item())
                _, val_pred = dev_out.max(dim=1)

                truth.append(dev_y.to('cpu').numpy()[0])
                prediction.append(val_pred.to('cpu').numpy()[0])
            # print(truth, prediction)
            acc = accuracy_score(truth, prediction)
            f1 = f1_score(truth, prediction, average=None)
            p = precision_score(truth, prediction, average=None, zero_division=True)
            r = recall_score(truth, prediction, average=None)
            print('acc:', acc, ', F1:', f1, ' ,Precision:', p, ' ,Recall:', r)
        th.save(model.state_dict(), 'check_points/model' + str(epoch) + '.pt')




if __name__ == '__main__':
    train()