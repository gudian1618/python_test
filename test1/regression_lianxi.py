import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

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

net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)   # 传入所有net参数，学习效率0.5
loss_func = torch.nn.MSELoss()  # 定义损失函数为均方误差函数

for t in range(200):
    prediction = net(x)   # 给net输入训练数据，输出预测值

    loss = loss_func(prediction, y) # 通过损失函数，计算预测值和真实值得误差

    optimizer.zero_grad()   # 清空每次迭代残余更新值
    loss.backward()     # 误差反向传播，计算参数更新值
    optimizer.step()    # 将参数更新值施加到net参数上
    if t % 5 == 0:      # 每学习5步进行绘图
        # 画出并显示出学习过程
        plt.cla()   # 每次刷新前清图
        plt.scatter(x.data.numpy(), y.data.numpy())     # 转换后画散点图
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss=%.4f' % loss.data[0],
                 fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


