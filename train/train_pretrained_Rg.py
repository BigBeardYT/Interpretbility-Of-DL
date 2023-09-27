import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from RgNormResNet_3.my_utils.load_datasets import load_datasets
from RgNormResNet_3.models.load_models import get_model

# from RgNormResNet.my_utils.set_random_seeds import setup_seed
#
# # 设置随机数种子
# random_seed = 1313
# setup_seed(random_seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def my_train(data_name, model_name, num_classes, train_loader, test_loader,
             batch_size, num_epochs, lr, start, end):
    save_name = data_name + '_' + model_name
    if data_name != 'CiFar10':
        in_features = 1
    else:
        in_features = 3

    for i in range(start, end):
        print("第{}次预训练, 模型:{}, 数据集:{}".format(i, model_name, data_name))
        # 加载预训练模型
        pretrained_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)

        pretrained_model_params_path = '../trained_model/' + 'Pretrained_' + save_name + '_bz' + str(batch_size) \
                                       + '_ep10' + '_lr' + str(lr) + '_seedNone' \
                                       + str(i) + '.pth'
        print('进行对抗训练的模型参数所在位置\n' + pretrained_model_params_path)

        # 加载模型参数
        pretrained_model.load_state_dict(torch.load(pretrained_model_params_path))
        # 加载优化器
        optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=lr, momentum=0.9)
        train_acc_lst, valid_acc_lst = [], []
        train_loss_lst, valid_loss_lst = [], []
        best_acc = 0.0
        for epoch in range(num_epochs):

            pretrained_model.train()

            for batch_idx, (features, targets) in enumerate(train_loader):

                ### PREPARE MINIBATCH
                features, targets = features.to(device), targets.to(device)

                ### FORWARD AND BACK PROP
                outputs = pretrained_model(features)
                predicts = F.softmax(outputs, dim=1)
                loss = F.cross_entropy(outputs, targets)
                optimizer.zero_grad()

                loss.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                if batch_idx % 75 == 0:
                    train_loss_lst.append(loss.item())

                ### LOGGING
                if not batch_idx % 200:
                    print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                          f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                          f' Loss: {loss:.4f}')

            # no need to build the computation graph for backprop when computing accuracy
            pretrained_model.eval()
            with torch.no_grad():
                # 训练精度、训练损失以及测试的精度和损失
                train_acc, train_loss = compute_accuracy_and_loss(pretrained_model, train_loader, device=device)
                valid_acc, valid_loss = compute_accuracy_and_loss(pretrained_model, test_loader, device=device)
                train_acc_lst.append(train_acc)
                valid_acc_lst.append(valid_acc)
                # train_loss_lst.append(train_loss)
                valid_loss_lst.append(valid_loss)
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} Train Acc.: {train_acc:.2f}%'
                      f' | Validation Acc.: {valid_acc:.2f}%')

            # 保存模型
            if valid_acc > best_acc:
                print('Saving Model...')
                best_acc = valid_acc
                # 最好模型参数的存储路径
                best_model_params_path = '../trained_model/' + save_name + '_bz' \
                                         + str(batch_size) + '_ep' + str(num_epochs) \
                                         + '_lr' + str(lr) + '_Pretrained' + '_seedNone' + str(i) + '.pth'
                torch.save(pretrained_model.state_dict(), best_model_params_path)

            # # 动态更改学习率
            # if (epoch + 1) == int(num_epochs * 0.5) or (epoch + 1) == int(num_epochs * 0.75) \
            #         or (epoch + 1) == int(num_epochs * 0.95):
            #     for params_group in optimizer.param_groups:
            #         params_group['lr'] *= 0.1
            #         print('更改学习率为{}:'.format(params_group['lr']))

        plt.plot(train_loss_lst, label='loss')
        plt.legend()
        plt.title(save_name + '_train_loss')
        # 存储图片
        plt.savefig('../saveimage/' + save_name + '_bz' + str(batch_size)
                    + '_ep' + str(num_epochs) + '_lr' + str(lr) + '_Pretrained'
                    + '_seedNone' + str(i) + '_trainloss' + '.png')
        plt.show()


# 计算精确度和损失
def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        # predicts = F.softmax(outputs, dim=1)
        cross_entropy += F.cross_entropy(outputs, targets).item()
        _, predicted_labels = torch.max(outputs, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


if __name__ == '__main__':
    data_name = 'CiFar10'
    model_name = 'RgResNet'
    num_classes = 10
    """ 超参数 """
    batch_size = 32
    num_epochs = 30
    lr = 0.01
    start = 1
    end = 3
    # 加载数据集
    train_dataset, test_dataset, \
    train_loader, test_loader = \
        load_datasets(batch_size=batch_size, data_name=data_name)

    my_train(data_name, model_name, num_classes, train_loader, test_loader,
             batch_size, num_epochs, lr, start, end)
