import torch
from RgNormResNet_3.my_utils.fgsm import *


def iter_on_advimage(model, images, adv_images, labels, criterion, device, epsilon=0.01, iters = 5, minv = 0, maxv = 1):
    # print("攻击时的迭代次数iters: {}".format(iters))
    for i in range(iters):
        adv_images_grad = get_gradient(model, adv_images, labels, criterion, device)

        # 使用 with torch.no_grad() 上下文管理器，这样可以防止在 adv_images 变量参与迭代时被跟踪,
        # 从而避免出现 RuntimeError: you can only change requires_grad flags of leaf variables 错误.
        with torch.no_grad():
            adv_images = adv_images + epsilon * adv_images_grad.sign()
            eta = torch.clamp(adv_images - images, min = -epsilon, max = epsilon)
            adv_images = torch.clamp(images + eta, min = minv, max = maxv)

            # 将下面这行代码移动到 adv_images 更新之后，确保能够正确计算梯度
            adv_images.requires_grad = True

    return eta, adv_images


def generate_bim_noise(model, images, labels, criterion, device,
                       epsilon=0.01, iters = 5, minv = 0, maxv = 1):
    """
    传入模型model，图像images，生成带有 BIM 对抗噪声的图片
    :param model: 模型
    :param images: 图像
    :param epsilon: 噪声系数
    :param iters: 迭代次数
    :return: 对抗噪声，对抗样本图片
    """
    adv_images = images.clone().detach().requires_grad_(True).to(device)
    adv_images.requires_grad = True

    return iter_on_advimage(model, images, adv_images, labels, criterion, device, epsilon, iters, minv = minv, maxv = maxv)


def generate_pgd_noise(model, images, labels, criterion, device,
                       epsilon=0.1, num_iter = 20, minv = 0, maxv = 1):
    # print("对抗攻击的迭代次数iters: {}".format(num_iter))
    """
    传入模型model，图像images，生成带有 PGD 对抗噪声的图片
    :param model: 模型
    :param images: 图像
    :param epsilon: 噪声系数
    :param iters: 迭代次数
    :return: 对抗噪声，对抗样本图片
    """

    # 先做一轮随机初始化，攻击效果会更好一点
    adv_images = images.clone().detach().requires_grad_(True).to(device)
    eta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adv_images = images + eta
    adv_images = torch.clamp(adv_images, min = minv, max = maxv).to(device)
    adv_images.requires_grad = True

    return iter_on_advimage(model, images, adv_images, labels, criterion, device, epsilon, num_iter, minv = minv, maxv = maxv)


def generate_CAA_noise(model, images, labels, criterion, device, epsilon = 0.1, num_iter = 5, minv = 0, maxv = 1):
    adv_images = images.clone().detach().requires_grad_(True).to(device)
    eta = torch.zeros_like(images).uniform_(-0.1, 0.1)
    adv_images = torch.clamp(images + eta, min = minv, max = maxv)
    # 组合了好几个攻击
    fgsm_adv_noise , fgsm_adv_images = generate_fgsm_noise(model = model,
                                                          images = adv_images,
                                                          labels = labels,
                                                          criterion = criterion,
                                                          device = device,
                                                          epsilon = 0.6 * epsilon,
                                                          minv = minv, maxv = maxv)
    eta, pgd_adv_images = generate_pgd_noise(model = model,
                                         images = fgsm_adv_images,
                                         labels = labels,
                                         criterion = criterion,
                                         device = device,
                                         epsilon = 0.5 * epsilon,
                                         num_iter = 2,
                                         minv = minv, maxv = maxv)
    eta, bim_adv_images = generate_bim_noise(model = model,
                                         images = pgd_adv_images,
                                         labels = labels,
                                         criterion = criterion,
                                         device = device,
                                         epsilon = 0.4 * epsilon,
                                         iters = 2,
                                         minv = minv, maxv = maxv)

    nifgsm_noise, nifgsm_adv_images = generate_ni_fgsm_noise(model, bim_adv_images.detach(), labels, criterion, device, 0.3 * epsilon, num_iter = 2, minv = minv, maxv = maxv)
    mifgsm_noise, mifgsm_adv_images = generate_mi_fgsm_noise(model, nifgsm_adv_images.detach(), labels, criterion, device, epsilon, num_iter =  2, minv = minv, maxv = maxv)
    return  mifgsm_noise, mifgsm_adv_images




