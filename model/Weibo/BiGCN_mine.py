import sys, os
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
import copy


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
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
        x = self.conv2(x, edge_index)
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
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
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
        x = self.conv2(x, edge_index)
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

        ## my code : self-attention
        self.fcQ = th.nn.Linear((out_feats + hid_feats) * 2, 1024)
        self.fcK = th.nn.Linear((out_feats + hid_feats) * 2, 1024)
        self.fcV = th.nn.Linear((out_feats + hid_feats) * 2, 1024)
        self.attention = th.nn.MultiheadAttention(1, 1)
        self.fc = th.nn.Linear(1024, 2)
    def forward(self, data):

        print(data)
        #print(data.x)
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        #print(x.shape)
        query = self.fcQ(x)
        query = query.permute(1, 0).unsqueeze(dim=-1)
        #print(query.shape)
        key = self.fcK(x)
        key = key.permute(1, 0).unsqueeze(dim=-1)
        value = self.fcV(x)
        value = value.permute(1, 0).unsqueeze(dim=-1)
        x, _ = self.attention(query,
                              key,
                              value)
        #print(x.shape)
        x = x.squeeze(dim=-1).permute(1, 0)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize,
              dataname, iter):
    model = Net(10000, 1024, 64).to(device)
    BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer1 = th.optim.Adam([
        {'params': base_params},
        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr / 5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr / 5}
    ], lr=lr, weight_decay=weight_decay)
    optimizer2 = th.optim.SGD(
        # params=filter(lambda p: p.requires_grad, model.parameters()),
        params=[
            {'params': base_params},
            {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr},
            {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr}
        ],
        momentum=0.9,
        lr=lr
    )

    optimizer = optimizer1
    model.train()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)

    traindata_list = [traindata_list[1], traindata_list[2]]

    for epoch in range(n_epochs):
        total_sample = 4600
        sweight = [2 if i['y'] == 0 else 1 for i in traindata_list]

        wsampler = WeightedRandomSampler(sweight, num_samples=total_sample, replacement=True)
        train_loader = DataLoader(traindata_list, batch_size=batchsize,
                                  shuffle=False, num_workers=10 )#sampler=wsampler)
        test_loader = DataLoader(testdata_list, batch_size=batchsize,
                                 shuffle=False, num_workers=10)
        model.train()
        avg_loss, avg_acc = [], []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels = model(Batch_data)
            #print(out_labels, Batch_data.y)
            loss = F.nll_loss(out_labels, Batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            postfix = "Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,
                                                                                                                   epoch,
                                                                                                                   batch_idx,
                                                                                                                   loss.item(),
                                                                                                                   train_acc)
            tqdm_train_loader.set_postfix_str(postfix)
            batch_idx = batch_idx + 1
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses, temp_val_accs, temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], [], [], [], [], [], [], [], []
        model.eval()

        with th.no_grad():
            tqdm_test_loader = tqdm(test_loader)
            for Batch_data in tqdm_test_loader:
                Batch_data.to(device)
                val_out = model(Batch_data)
                val_loss = F.nll_loss(val_out, Batch_data.y)
                temp_val_losses.append(val_loss.item())
                _, val_pred = val_out.max(dim=1)
                correct = val_pred.eq(Batch_data.y).sum().item()
                #print(val_pred, Batch_data.y)
                val_acc = correct / len(Batch_data.y)
                Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                    val_pred, Batch_data.y)
                temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                    Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                    Recll2), temp_val_F2.append(F2)
                temp_val_accs.append(val_acc)
            val_losses.append(np.mean(temp_val_losses))
            val_accs.append(np.mean(temp_val_accs))
            print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                               np.mean(temp_val_accs)))

            res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
                   'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                           np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
                   'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                           np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
            print('results:', res)


    return
    # return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2


if __name__ == "__main__":
    lr = 0.0005
    weight_decay = 1e-4
    patience = 5
    n_epochs = 200
    batchsize = 128
    tddroprate = 0.1
    budroprate = 0.1
    datasetname = "Weibo"
    iterations = int(sys.argv[1])
    model = "BiGCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], [], []

    x_train = load_train(datasetname)   # x_train 全是str_id

    x_test = load_test(datasetname)     # x_test 也是对应的str_id
    iter = 1  # useless now
    treeDic = loadTree(datasetname)
    train_GCN(treeDic,
              x_test,
              x_train,
              tddroprate, budroprate,
              lr, weight_decay,
              patience,
              n_epochs,
              batchsize,
              datasetname,
              iter)
