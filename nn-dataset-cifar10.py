'''
    2020.03.21
    数据集：cifar-10
    训练集： 5个文件就是分成了5份的那50000张图片:
            data_batch_1.bin、data_batch_2.bin、data_batch_3.bin、data_batch_4.bin、data_batch_5.bin
    测试集：总共10000张图片: test_batch.bin
    类别说明：说明了整个cifar-10数据集所包括的10个事物类别: batches.meta.txt

    网络模型：LeNet（自定义） ---  Accuracy：64%
             ResNet34（加载已下载的预训练模型） ---  Accuracy：82%
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# cifar-10官方提供的数据集是用numpy array存储的
# 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 在构建数据集的时候指定transform，就会应用我们定义好的transform
# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          transform=transform)

trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True)


class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# GPU
device = torch.device("cuda:0")
resnet34 = torchvision.models.resnet34(pretrained=False)
resnet34.load_state_dict(torch.load('data/resnet34-333f7ec4.pth'))
# net = LeNet().to(device)
net = resnet34.to(device)


# print(
#     '\nLenet:', LeNet(),
#     '\nresnet:', resnet34,
# )


# CrossEntropyLoss就是我们需要的损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 下面我们正式开始训练
print("Start Training...")
for epoch in range(30):
    # 我们用一个变量来记录每100个batch的平均loss
    loss100 = 0.0
    # 我们的dataloader派上了用场
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 注意需要复制到GPU
        optimizer.zero_grad()  # 首先要把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
        outputs = net(inputs)  # 计算前向传播的输出
        loss = criterion(outputs, labels)  # 根据输出计算loss
        loss.backward()  # 算完loss之后进行反向梯度传播，这个过程之后梯度会记录在变量中
        optimizer.step()  # 用计算的梯度去做优化
        loss100 += loss.item()
        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss100 / 100))
            loss100 = 0.0

print("Done Training!")


# 用训练好的模型来预测test数据集
# 构造测试的dataloader
dataiter = iter(testloader)
# 预测正确的数量和总数量
correct = 0
total = 0
# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 预测
        outputs = net(images)
        # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))