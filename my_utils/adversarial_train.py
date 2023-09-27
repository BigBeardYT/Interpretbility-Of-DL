import sys

import torch.nn

from RgNormResNet_3.my_utils.fgsm import generate_fgsm_noise

sys.path.append("..")
import torch.nn as nn
from RgNormResNet_3.my_utils.pgd import *
from RgNormResNet_3.my_utils.fgsm import *
from RgNormResNet_3.my_utils.load_models import get_model
from RgNormResNet_3.my_utils.trades import trades_loss
# 导入datetime模块
from datetime import datetime

# 获取当前日期和时间
now = datetime.now()

# 输出当前日期和时间
print("当前日期为: {}, 时间: {}".format(now.date(), now.strftime("%H:%M:%S")))

device = 'cuda'
# epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
epsilons = [2 / 255]
# 损失函数
criterion = torch.nn.CrossEntropyLoss()

adv_train_epochs = 10


def trades_adv_train(data_name, model_name,
                     test_loader, num_classes, lr,
                     batch_size, num_epochs,
                     start, end):
    print("TRADES对抗训练")

    save_name = data_name + '_' + model_name + '_' + 'Trades' + '_train'
    # 输入通道
    in_features = 3
    if data_name != 'CiFar10' and data_name != 'SVHN':
        in_features = 1

    for i in range(start, end):
        # 模型对抗攻击
        print('第{}次对抗训练, 模型: {}, 数据集: {}, 对抗训练方法: {}'.format(
            i, model_name, data_name, 'Trades'))

        # 加载模型
        attacked_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
        # print('模型:', attacked_model)
        # 路径
        attacked_model_params_path = '../trades_trained_model/' + save_name + '_bz' + str(batch_size) \
                                     + '_ep' + str(num_epochs) + '_lr' + str(lr) + '_seedNone' \
                                     + str(i) + '.pth'
        # attacked_model_params_path = '../trained_model/' + 'The_Best_RgNormAlex_bz32_epochs30.pth'
        print('进行对抗训练的模型参数所在位置\n' + attacked_model_params_path)

        # 加载模型参数
        attacked_model.load_state_dict(torch.load(attacked_model_params_path))

        optimizer = torch.optim.SGD(attacked_model.parameters(), lr=lr, momentum=0.9)
        # 对抗训练的损失值、 精度
        adv_train_loss_lst = []
        adv_train_acc_lst = []

        total_correct = 0
        total_samples = 0
        best_acc = 0.0
        for epoch in range(adv_train_epochs):
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = attacked_model(images)
                # trades对抗训练，在训练时将损失函数换成trades_loss
                loss = trades_loss(attacked_model,
                                   images,
                                   labels,
                                   optimizer,
                                   step_size=0.0001,
                                   epsilon=0.0078,  # 2/255
                                   perturb_steps=10,
                                   beta=1.0,
                                   distance='l_inf')
                # 反向传播

                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 200 == 0:
                    print('Epochs: [{}/{}],Step: [{}/{}], Adv_Train_Way: {}, Loss: {:.4f}'.format(
                        epoch + 1, adv_train_epochs, batch_idx + 1, len(test_loader), 'Trades',
                        loss.item()))
                adv_train_loss_lst.append(loss.item())
                # 准确率
                _, pred = torch.max(outputs, 1)
                total_correct += torch.sum(pred == labels)
                adv_train_acc_lst.append(total_correct / total_samples)

            attack_train_acc = (total_correct / total_samples) * 100
            print('Epochs: [{}/{}], Adv_Train_Way: {}, Accuracy: {:.2f}'.format(epoch+1, adv_train_epochs,
                'Trades', attack_train_acc.item()))

            # 保存模型
            if attack_train_acc > best_acc:
                print('Saving Model...')
                best_acc = attack_train_acc
                # 重新存储参数的路径
                best_model_params_path = '../trades_trained_model/' + save_name + '_bz' + str(batch_size) \
                                         + '_ep' + str(num_epochs) + '_lr' + str(lr) + '_seedNone' \
                                         + str(i) + '.pth'
                torch.save(attacked_model.state_dict(), best_model_params_path)


def noise_train(noise_name, data_name, model_name,
                test_loader, num_classes, lr,
                batch_size, num_epochs,
                start, end):
    if noise_name == 'PGD':
        print('PGD-普通-对抗训练...')
    elif noise_name == 'BIM':
        print('BIM-普通-对抗训练...')
    elif noise_name == 'FGSM':
        print('FGSM-普通-对抗训练...')
    else:
        print('输入的攻击方法有误!!!')
        return

    save_name = data_name + '_' + model_name + '_' + 'PGD' + '_train'
    # 输入通道
    in_features = 3
    if data_name != 'CiFar10' and data_name != 'SVHN':
        in_features = 1

    for i in range(start, end):
        # 模型对抗攻击
        print('第{}次对抗训练, 模型 {}, 数据集 {}, 对抗训练方法 {}'.format(
            i, model_name, data_name, noise_name))

        for epsilon in epsilons:
            # 加载模型
            attacked_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
            # print('模型:', attacked_model)
            # 路径
            attacked_model_params_path = '../trained_model/' + save_name + '_bz' + str(batch_size) \
                                         + '_ep' + str(num_epochs) + '_lr' + str(lr) + '_seedNone' \
                                         + str(i) + '.pth'
            # attacked_model_params_path = '../trained_model/' + 'The_Best_RgNormAlex_bz32_epochs30.pth'
            print('进行对抗训练的模型参数所在位置\n' + attacked_model_params_path)

            # 加载模型参数
            attacked_model.load_state_dict(torch.load(attacked_model_params_path))
            optimizer = torch.optim.SGD(attacked_model.parameters(), lr=lr, momentum=0.9)
            # 对抗训练的损失值、 精度
            adv_train_loss_lst = []
            adv_train_acc_lst = []
            criterion = nn.CrossEntropyLoss()
            total_correct = 0
            total_samples = 0
            best_acc = 0.0
            for epoch in range(adv_train_epochs):
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)
                    # 原始输出结果
                    init_outputs = attacked_model(images)
                    # 预测结果
                    _, pred = torch.max(init_outputs, 1)
                    # 样本总数
                    total_samples += init_outputs.shape[0]
                    # 生成PGD噪声
                    if noise_name == 'PGD':
                        iters = generate_pgd_noise(attacked_model, images, labels, criterion, device,
                                                   epsilon=epsilon, num_iter=10, minv=0, maxv=1)
                    elif noise_name == 'BIM':
                        iters = generate_bim_noise(attacked_model, images, labels, criterion, device,
                                                   epsilon=epsilon, iters=5, minv=0, maxv=1)
                    elif noise_name == 'FGSM':
                        iters = generate_fgsm_noise(attacked_model, images, labels, criterion, device,
                                                    epsilon=epsilon, minv=0, maxv=1)
                    # 攻击之后的样本
                    eta, adv_images = iters
                    # 将 adv_images进行重新训练
                    outputs = attacked_model(adv_images)
                    loss = criterion(outputs, labels)
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (batch_idx + 1) % 200 == 0:
                        print('Epoch: [{}/{}], Step: [{}/{}], Epsilon: {}, Noise: {}, Loss: {:.4f}'.format(
                            epoch + 1, adv_train_epochs, batch_idx + 1, len(test_loader), epsilon, noise_name,
                            loss.item()))
                    adv_train_loss_lst.append(loss.item())
                    # 准确率
                    _, pred = torch.max(outputs, 1)
                    total_correct += torch.sum(pred == labels)
                    adv_train_acc_lst.append(total_correct / total_samples)

                attack_train_acc = (total_correct / total_samples) * 100
                print('Epoch: [{}/{}], Epsilon: {}, Noise: {}, Accuracy: {:.2f}'.format(
                    epoch + 1, adv_train_epochs, epsilon, noise_name, attack_train_acc.item()))

                # 保存模型
                if attack_train_acc > best_acc:
                    print('Saving Model...')
                    best_acc = attack_train_acc
                    # 重新存储参数的路径
                    best_model_params_path = '../trained_model/' + save_name + '_bz' + str(batch_size) \
                                             + '_ep' + str(num_epochs) + '_lr' + str(lr) + '_seedNone' \
                                             + str(i) + '.pth'
                    torch.save(attacked_model.state_dict(), best_model_params_path)
