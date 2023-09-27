""" 根据函数传入的实参，返回相应的模型 """
import sys
sys.path.append("..")
from RgNormResNet_3.models.resnet import resnet18
from RgNormResNet_3.models.rgresnet import RgResNet18
from RgNormResNet_3.my_utils.resnet import SparseResNet18
from RgNormResNet_3.models.rg_wideresnet import RgWideResNet
from RgNormResNet_3.models.resnet_fsr import ResNet18_FSR
# from RgNormResNet_2.models.rgresnet import RgResNet18
# from RgNormResNet.models.test_rgmodel import RgResNet18


def get_model(model_name, in_features=1, num_classes=10):
    """ 传入模型名称，以及分类数 """
    if model_name == 'ResNet':
        return resnet18(in_features=in_features, num_classes=num_classes)
    elif model_name == 'RgResNet':
        return RgResNet18(in_features=in_features, num_classes=num_classes)
    elif model_name == 'SpResNet':
        return SparseResNet18(sparsities=[0.1, 0.1, 0.1, 0.1], sparse_func='vol')
    elif model_name == 'RgWideResNet':
        return RgWideResNet()
    elif model_name == 'FSRResNet':
        return ResNet18_FSR(in_features=in_features)
    else:
        print("输入的模型有误!!!")
        return None
