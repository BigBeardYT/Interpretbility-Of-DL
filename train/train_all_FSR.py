import sys
import warnings

sys.path.append("..")
from RgNormResNet_3.my_utils.load_datasets import load_datasets
from RgNormResNet_3.my_utils.train_implement import my_train

FSR_ResNet_model_name = 'FSRResNet'

""" ######## 以下参数训练之前手动设置 ######### """
batch_size = 32
num_epochs = 10
lr = 0.01
data_name = 'KMNIST'
model_name = FSR_ResNet_model_name
""" ######## 以上参数训练之前手动设置 ######### """

# 不同的数据集不同的分类，这五个数据集中，只有EMNIST是37分类，其他 都为10分类
num_classes = 10

# 加载数据集
train_dataset, test_dataset, \
train_loader, test_loader = \
    load_datasets(batch_size=batch_size, data_name=data_name)

my_train(data_name, model_name, num_classes,
         train_loader, test_loader,
         batch_size, num_epochs, lr, 1, 4)

