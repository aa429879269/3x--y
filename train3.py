#GRU 16X16X16X16

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn import init
import logging
precent = 0.8
torch.manual_seed(1)
device = torch.device("cuda:0")

dt = pd.read_csv("22sensordata.csv")
data_all = dt.values
np.random.shuffle(dt.values)
#20~228
#24~95
#25~91
#2400~46900


train_X = data_all[:int(np.size(data_all,0)*precent), 0:3]
train_Y = data_all[:int(np.size(data_all,0)*precent), 3]
test_X = data_all[int(np.size(data_all,0)*precent):, 0:3]
test_Y = (data_all[int(np.size(data_all,0)*precent):, 3])*45000+2000
x1 = (test_X[:,0])*215+15
x2 = (test_X[:,1])*80+20
x3 =  (test_X[:,2]) * 45000 + 2000

train_X = torch.Tensor(train_X).to(device)
train_Y = torch.Tensor(train_Y).unsqueeze(1).to(device)
test_X = torch.Tensor(test_X).to(device)
test_Y = torch.Tensor(test_Y).unsqueeze(1).to(device)

train_label=[]
for i in range(len(train_X)):
    train_label.append([train_X[i],train_Y[i]   ])

class MyData(torch.utils.data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]

data = MyData(train_label)

data_loader = DataLoader(data, batch_size=train_X.size(0), shuffle=True)


class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2,n_hidden_3,out_dim=1):
        super(Activation_Net, self).__init__()
        self.gru = nn.GRU(1,16,1,bias=True,batch_first=True).to("cuda:0")
        self.layer1 = nn.Linear(in_dim, n_hidden_1,bias=True).to("cuda:0")
        #self.m1 = nn.BatchNorm1d(16).to("cuda:0")
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2,bias=True).to("cuda:0")
        #self.m2 = nn.BatchNorm1d(16).to("cuda:0")
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3,bias=True).to("cuda:0")
        #self.m3 = nn.BatchNorm1d(16).to("cuda:0")
        self.layer4 = nn.Linear(n_hidden_3, out_dim, bias=True).to("cuda:0")
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """
    def init_p(self):
        init.xavier_normal_(self.layer1.weight)
        init.xavier_normal_(self.layer2.weight)
        init.xavier_normal_(self.layer3.weight)
        init.xavier_normal_(self.layer4.weight)
    def forward(self, x):
        output, hn = self.gru(x)
        hn = hn.view(-1,16)
        x = F.relu(self.layer1(hn))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
def evalute(model):
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        train_logits = model.forward(train_X.unsqueeze(2))
        tarin_loss = loss_fn(train_logits, train_Y)
        val_logits = (model.forward(test_X.unsqueeze(2)))*45000+2000
        val_loss = loss_fn(val_logits, test_Y)
    model.train()
    return train_logits,tarin_loss,val_logits,val_loss



def train(epoch):
    model = Activation_Net(16,16,16,16)
    model.init_p()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for e in range(epoch):  # batch_idx是enumerate（）函数自带的索引，从0开始
        for batch_id,(x,y) in enumerate(data_loader):
            x = x.unsqueeze(2)
            optimizer.zero_grad()  # 所有参数的梯度清零
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()  # 即反向传播求梯度
            optimizer.step()
        if e%1000==0:
            train_logit,train_loss,val_logit,val_loss = evalute(model)
            train_big  = max(abs(train_logit - train_Y))
            val_big = max(abs(val_logit-test_Y))
            print(e, train_big.cpu().numpy(), train_loss.cpu().numpy(), val_big.cpu().numpy(), val_loss.cpu().numpy())
            if e % 20000==0:
                output = pd.DataFrame(test_X.cpu().numpy())
                output[3] = np.squeeze(test_Y.cpu().numpy())    #测试结果
                output[4] = np.squeeze(val_logit.cpu().numpy())     #验证集
                output[5] = abs(output[3] - output[4])      #测试集误差
                output[6] = x1      #
                output[7] = x2
                output[8] = x3
                output.to_csv('data_test/csv/eval_{}.csv'.format(e))
                torch.save(model, 'data_test/model/model_{}.pkl'.format(e))
    return model


train(1000000)


