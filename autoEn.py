import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import sys

sys.path.append('F:\\PINN\\Physics-Informed-Neural-Networks-Multitask-Learning-master\\AE\\')
from AutomaticWeightedLoss import AutomaticWeightedLoss


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 96),
            nn.Tanh(),
            nn.Linear(96, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 24),
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 96),
            nn.Tanh(),
            nn.Linear(96, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 7),
            nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 数据读取
import pandas as pd

df = pd.read_csv('F:\\PINN\\Physics-Informed-Neural-Networks-Multitask-Learning-master\\AE\\121.csv')
df = df.drop(['Unnamed: 0'], axis=1)
X_train = df[:999]
X_test = df[1500:1999]
##########

###########
# 数据类型转换
trainData = torch.FloatTensor(X_train.values)
testData = torch.FloatTensor(X_test.values)
#####################################
# 归一化 #
# mu = torch.mean(trainData, dim=0)
# std = torch.std(trainData, dim=0)
# trainData1 = (trainData-mu)/std
# testData1 = (testData-mu)/std
# or #
mx = torch.max(trainData, 0)
mn = torch.min(trainData, 0)
mx1 = mx.values
mn1 = mn.values
trainData1 = (trainData - mn1) / (mx1 - mn1)
testData1 = (testData - mn1) / (mx1 - mn1)
######################################

# 构建张量数据集
train_dataset = TensorDataset(trainData1, trainData1)
test_dataset = TensorDataset(testData1, testData1)
trainDataLoader = DataLoader(dataset=train_dataset, batch_size=200)

########################################

# 初始化
epochs = 1000
autoencoder = AutoEncoder()
print(autoencoder)
awl = AutomaticWeightedLoss(2)  ## we have 2 losses
loss_1 = nn.MSELoss()
loss_2 = nn.MSELoss()

optimizer = optim.Adam([
    {'params': autoencoder.parameters(), 'lr': 0.001},
    {'params': awl.parameters(), 'weight_decay': 0}
])

# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, cooldown=0, min_lr=0, eps=1e-08)
loss_ae = np.zeros((epochs, 1))
loss_ph = np.zeros((epochs, 1))
loss_train = np.zeros((epochs, 1))
for epoch in range(epochs):
    # 不需要label，所以用一个占位符"_"代替
    for batchidx, (x, _) in enumerate(trainDataLoader):
        # 编码和解码
        encoded, decoded = autoencoder(x)
        # 计算loss
        # loss = loss_func(decoded, x)
        loss1 = loss_1(decoded, x)
        #loss2 = loss_2((torch.mean(torch.sum((decoded * (mx1 - mn1) + mn1)[:, :5], dim=1))-torch.mean(mn1))/torch.mean((mx1 - mn1)),
        #(torch.mean((decoded * (mx1 - mn1) + mn1)[:, 6])-torch.mean(mn1))/torch.mean((mx1 - mn1)))/5000
        #loss2 = loss1
        loss2 = loss_2(torch.mul((testData1 * (mx1 - mn1) + mn1)[:, 5],
                                 torch.sub((testData1 * (mx1 - mn1) + mn1)[:, 3], (testData1 * (mx1 - mn1) + mn1)[:, 2]))*500/3413,
                       torch.add(torch.mul((testData1 * (mx1 - mn1) + mn1)[:, 6],
                                 torch.sub((testData1 * (mx1 - mn1) + mn1)[:, 0], (testData1 * (mx1 - mn1) + mn1)[:, 1]))*500/3413,
                                 (testData1 * (mx1 - mn1) + mn1)[:, 4] )
                       )

        loss_sum = awl(loss1, loss2)
        # 更新
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        # scheduler.step(loss1)
        loss_ae[epoch, 0] = loss1.item()
        loss_ph[epoch, 0] = loss2.item()
        loss_train[epoch, 0] = loss_sum.item()
        print('Epoch: %04d, Training loss=%.8f' %
              (epoch + 1, loss_sum.item()))
###########################################
fig = plt.figure(figsize=(6, 3))
ax = plt.subplot(2, 1, 1)
ax.grid()
ax.plot(loss_ae, color=[245 / 255, 124 / 255, 0 / 255], linestyle='-', linewidth=2)
ax.set_xlabel('Epoches')
ax.set_ylabel('AELoss')
ax = plt.subplot(2, 1, 2)
ax.grid()
ax.plot(loss_ph, color=[0 / 255, 124 / 255, 245 / 255], linestyle='-', linewidth=2)
ax.set_xlabel('Epoches')
ax.set_ylabel('PHLoss')

plt.show()

##############################################
_, decodedTestdata = autoencoder(testData1)
decodedTestdata = decodedTestdata.double()
reconstructedData = decodedTestdata.detach().numpy()
testData2 = testData1.double()
testData2 = testData2.detach().numpy()
############################################
# fig = plt.figure(figsize=(6, 9))
# for i in range(X_test.values.shape[1]):
#     ax = plt.subplot(7, 1, i+1)
#     ax.plot(X_test.values[:, i], color='C0', linestyle='-', linewidth=2)
#     ax.plot(reconstructedData[:, i], color='C3', linestyle='-', linewidth=2)
#     ax.legend(['Real value','Reconstructed value'], loc="upper right",
#           edgecolor='black', fancybox=True)
# plt.show()
aa = torch.sum(testData1[:, :5], dim=1) - testData1[:, 6]
aaa = X_test.values.shape[1]
###############################
fig = plt.figure(figsize=(6, 9))
for i in range(aaa):
    ax = plt.subplot(7, 1, i + 1)
    ax.plot(testData2[:, i], color='C0', linestyle='-', linewidth=2)
    ax.plot(reconstructedData[:, i], color='C3', linestyle='-', linewidth=2)
    ax.legend(['Real value', 'Reconstructed value'], loc="upper right",
              edgecolor='black', fancybox=True)
plt.show()

rmse = mean_squared_error(reconstructedData, testData2, squared=False)
