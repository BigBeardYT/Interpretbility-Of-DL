""" 一个专门封装了所有训练的类 """
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# from RgNormResNet_3.my_utils.set_random_seeds import setup_seed
#
# # 设置随机数种子
# random_seed = 42
# setup_seed(random_seed)

class Train:
    def __init__(self, train_loader, test_loader, train_model, batch_size=32, lr=0.001, num_epochs=30,
                 train_loss=[], seed=None, device='cuda', save_name=None):

        print('数据、参数等初始化中...')

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = train_model.to(device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.save_name = save_name
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(train_model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)
        self.num_epochs = num_epochs
        self.device = device

        self.train_loss = train_loss
        self.best_acc = 0.0
        self.seed = seed

    def train(self):
        print('-' * 10 + " 开始训练模型 " + '-' * 10)
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 梯度清零与反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 存储损失值，数据可视化
                if (i + 1) % 50 == 0:
                    self.train_loss.append(loss.item())
                # 打印训练过程
                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], train_loss: {:.4f}'.format(
                        epoch + 1, self.num_epochs, i + 1, len(self.train_loader), loss.item()
                    ))

            # 动态更改学习率
            if (epoch + 1) == int(self.num_epochs * 0.55) or (epoch + 1) == int(self.num_epochs * 0.80) \
                    or (epoch + 1) == int(self.num_epochs * 0.95):
                for params_group in self.optimizer.param_groups:
                    params_group['lr'] *= 0.1

            # 验证模型
            self.model.eval()
            total_correct = 0.0
            total_samples = 0.0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    test_outputs = self.model(images)
                    _, pred = torch.max(test_outputs, 1)

                    total_correct += torch.sum(pred == labels)
                    total_samples += test_outputs.shape[0]

                # 计算验证的精度
                test_acc = total_correct / total_samples
                # 打印结果
                print('Epoch [{}/{}], Test_Acc: {:.4f}'.format(epoch + 1, self.num_epochs, test_acc))
                # 与最好的比较，并保存
                if self.best_acc < test_acc:
                    print('saving model params ...')
                    self.best_acc = test_acc
                    # 最好模型参数的存储路径
                    best_model_params_path = '../trained_model/' + self.save_name + '_bz' \
                                             + str(self.batch_size) + '_ep' + str(self.num_epochs) \
                                             + '_lr' + str(self.lr) + '_seed' + str(self.seed) + '.pth'
                    torch.save(self.model.state_dict(), best_model_params_path)

        # 全部的epoch都训练完，进行数据可视化
        # plt.plot(self.train_loss, label='loss')
        # plt.legend()
        # plt.title(self.save_name + 'train_loss')
        # # 存储图片
        # plt.savefig('../saveimage/' + self.save_name + '_bz' + str(self.batch_size)
        #             + '_ep' + str(self.num_epochs) + '_lr' + str(self.lr)
        #             + '_seed' + str(self.seed) + '_trainloss' + '.png')
        # plt.show()
