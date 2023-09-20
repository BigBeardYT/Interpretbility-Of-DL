import sys

from RgNormResNet_3.my_utils.adversarial_train import noise_train, trades_adv_train

sys.path.append("/home/aigroup/yt_project/project/")
from RgNormResNet_3.my_utils.load_datasets import load_datasets
from RgNormResNet_3.my_utils.train_implement import my_train

# 模型（三通道的图，应该加载不同的模型）
ResNet_model_name = 'ResNet'

""" ######## 以下参数训练之前手动设置 ######### """
attacked_batch_size = 128
attacked_num_epochs = 100
lr = 0.01
data_name = 'SVHN'
model_name = ResNet_model_name

""" ######## 以上参数训练之前手动设置 ######### """

# 不同的数据集不同的分类，这五个数据集中，只有EMNIST是37分类，其他都为10分类
num_classes = 10
# 攻击时使用的数据集大小
attack_used_batch_size = 64

# 加载数据集
train_dataset, test_dataset, \
train_loader, test_loader = \
    load_datasets(batch_size=attack_used_batch_size, data_name=data_name)


# trades_adv_train(data_name, model_name,
#                      test_loader, num_classes, lr,
#                      attacked_batch_size, attacked_num_epochs,
#                      1, 2)

noise_name = 'PGD'
noise_train(noise_name, data_name, model_name,
            test_loader, num_classes, lr,
            attacked_batch_size, attacked_num_epochs,
            1, 2)


