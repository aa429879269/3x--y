import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

precent = 80




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dt = pd.read_csv("all.csv")
data = dt.values
np.random.shuffle(dt.values)


train_X = data[:np.size(data,0)//precent, 0:3] /200
train_Y = data[:np.size(data,0)//precent, 3] / 50000
test_X = data[np.size(data,0)//precent:, 0:3] / 200
test_Y = data[np.size(data,0)//precent:, 3] / 50000

# print(X)
# print(Y)



train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y).unsqueeze(1)
test_X = torch.Tensor(test_X)
test_Y = torch.Tensor(test_Y).unsqueeze(1)


class Batch_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def train(epoch, arge):
    model = Activation_Net(3, arge[0], arge[1], 1)
    loss_fn = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    losses = []

    for e in range(epoch):  # batch_idx是enumerate（）函数自带的索引，从0开始
        output = model(train_X)

        loss = loss_fn(output, train_Y)

        losses.append(loss.item())

        optimizer.zero_grad()  # 所有参数的梯度清零
        loss.backward()  # 即反向传播求梯度
        optimizer.step()
        # if e % 1000 == 0:
        #     print(loss.item())

    return model


def evalute(model):
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        logits = model.forward(test_X)
        loss = loss_fn(logits, test_Y)

    return loss, logits


for l1 in range(5, 40, 5):
    for l2 in range(5, 40, 5):
        model = train(100000, (l1, l2))
        torch.save(model, 'model/model_{}_{}.pkl'.format(l1, l2))
        acc, out = evalute(model)
        print('{}_{} {}'.format(l1, l2, acc))
        e = pd.DataFrame(test_X.numpy()*200)
        e[3] = np.squeeze(test_Y.numpy())*50000
        e[4] = np.squeeze(out.numpy())*50000
        e.to_csv('data/eval_{}_{}.csv'.format(l1, l2))
