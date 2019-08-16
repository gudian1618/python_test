import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 生成数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], y.data.numpy()[:, 1],
# 			c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(2, 10, 2)
print(net)

plt.ion()  # 光滑绘图
plt.show()


optimizer = torch.optim.SGD(net.parameters(), lr=0.02)   # 传入所有net参数，学习效率0.02减慢
loss_func = torch.nn.CrossEntropyLoss()  # 定义损失函数为交叉熵误差函数

for t in range(70):
    out = net(x)
    

    loss = loss_func(out, y)  # 通过损失函数，计算预测值和真实值得误差

    optimizer.zero_grad()   # 清空每次迭代残余更新值
    loss.backward()     # 误差反向传播，计算参数更新值
    optimizer.step()    #
    if t % 2 == 0:      # 每学习2步进行绘图
        # 画出并显示出学习过程
        plt.cla()   # 每次刷新前清图
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y,
					s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200   # 预测值值中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy,
			 	fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# plt.off()      # 停止绘图
plt.show()
