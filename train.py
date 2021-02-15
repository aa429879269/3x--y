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
from tensorboardX import SummaryWriter
from Layer import LinearNorm
precent = 0.8
torch.manual_seed(1)
device = torch.device("cuda:0")

dt = pd.read_csv("2sensordata_已归一化.csv")
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
write = SummaryWriter(comment='Net1')

class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """
    def __init__(self):
        super(Activation_Net, self).__init__()
        self.gru = nn.GRU(1,16,1,bias=True,batch_first=True,bidirectional=False).to("cuda:0")
        linear_list = []
        linear_in = linear_layer = nn.Sequential(
                LinearNorm(3, 16, bias=True, w_init_gain='linear').to("cuda:0"),
                nn.ReLU(True),
            )
        linear_list.append(linear_in)
        for _ in range(9):
            linear_layer = nn.Sequential(
                LinearNorm(16, 16, bias=True, w_init_gain='linear').to("cuda:0"),
                nn.ReLU(True),
            )
            linear_list.append(linear_layer)
        linear_out = nn.Sequential(
            LinearNorm(16, 1, bias=True, w_init_gain='linear').to("cuda:0"),
        )
        linear_list.append(linear_out)
        self.linears = nn.ModuleList(linear_list)
        # self.linear_1 = LinearNorm(3, 16, bias=True, w_init_gain='linear').to("cuda:0")
        # self.linear_2 = LinearNorm(16, 16, bias=True, w_init_gain='linear').to("cuda:0")
        # self.linear_3 = LinearNorm(16, 1, bias=True, w_init_gain='linear').to("cuda:0")
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """
    def forward(self, hn):
        self.gru.flatten_parameters()
        output, hn = self.gru(hn)
        # # output = output.contiguous()
        # # output = output.view(-1,48)
        hn = hn.view(-1,16)
        for linear in self.linears:
            hn =  linear(hn)

        return hn

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
    model = Activation_Net()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005,momentum=0.5)
    for e in range(epoch):  # batch_idx是enumerate（）函数自带的索引，从0开始
        for batch_id,(x,y) in enumerate(data_loader):
            x = x.unsqueeze(2)
            optimizer.zero_grad()  # 所有参数的梯度清零
            output = model(x)
            loss = loss_fn(output, y)
            write.add_scalar('Train',loss,e)
            loss.backward()  # 即反向传播求梯度
            optimizer.step()
        if e%100==0:
            train_logit,train_loss,val_logit,val_loss = evalute(model)
            train_big  = max(abs(train_logit - train_Y))
            val_big = max(abs(val_logit-test_Y))
            print(e,train_loss,train_big,val_big, val_loss.cpu().numpy())
            if e % 4000==0:
                output = pd.DataFrame(test_X.cpu().numpy())
                output[3] = np.squeeze(test_Y.cpu().numpy())
                output[4] = np.squeeze(val_logit.cpu().numpy())
                output[5] = abs(output[3] - output[4])
                output[6] = x1
                output[7] = x2
                output[8] = x3
                output.to_csv('test/eval_4_{}.gru.csv'.format(e))
                #torch.save(model, 'data_GRU16X16X16X16_2289/model/model_{}.pkl'.format(e))

    write.add_graph(model,(x))
    write.close()
    return model


train(100000)


