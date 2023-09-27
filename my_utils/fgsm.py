import torch


# 获取反向传播生成的梯度
def get_gradient(model, images, labels, criterion, device):
    """
    获取图像的梯度
    :param model: 模型
    :param images: 图像
    :param labels: 图像标签
    :param device: 运算设备
    :param criterion: 损失函数
    :return: 图像的梯度
    """
    # 将模型设置为评估模式

    model.zero_grad()
    images.requires_grad = True

    # 将输入图像和标签移动到同一设备上
    images = images.to(device)
    labels = labels.to(device)

    # 前向传递图像，计算损失并计算梯度
    outputs = model(images)
    if isinstance(outputs, tuple):
        outputs = outputs.logits
    loss = criterion(outputs, labels)

    loss.backward()

    # 返回图像的梯度
    return images.grad.data


def generate_fgsm_noise(model, images, labels, criterion, device, epsilon=0.01, minv = 0, maxv = 1):
    """
    传入模型model，图像images，生成带有 FGSM 对抗噪声的图片
    :param model: 模型
    :param images: 图像
    :param labels: 图像标签
    :param device: 运算设备
    :param criterion: 损失函数
    :param epsilon: 噪声系数
    :return: 对抗噪声, 带有 FGSM 对抗噪声的图片
    """
    # 获取图像的梯度
    images_grad = get_gradient(model, images, labels, criterion, device)

    # 计算带有对抗扰动的图像，并将其剪裁到0和1之间
    adv_noise = epsilon * torch.sign(images_grad)
    adv_images = images + adv_noise
    adv_images = torch.clamp(adv_images, min = minv, max = maxv).detach()

    # 返回对抗扰动和标签列表
    return adv_noise, adv_images


# PGD 相较于 MI-FGSM 在白盒攻击层面来说要强，但是黑盒攻击层面来说要弱
def generate_mi_fgsm_noise(model, images, labels, criterion, device, epsilon=0.01, miu = 1.0, num_iter = 5, minv = 0, maxv = 1):
    """
    传入模型model，图像images，生成带有 MI-FGSM 对抗噪声的图片
    :param model: 模型
    :param images: 图像
    :param labels: 图像标签
    :param device: 运算设备
    :param criterion: 损失函数
    :param epsilon: 噪声系数
    :param miu: 衰减因子, 文章指出等于 1 效果比较好
    :param num_iter: 迭代次数
    :return: 对抗噪声, 带有 MI-FGSM 对抗噪声的图片
    """
    alpha = epsilon / num_iter
    # 获取图像的梯度
    images_grad = torch.zeros_like(images)
    adv_images = images.clone()
    for i in range(num_iter):
        """
        正规写法: 每次迭代算出梯度及其图像一范数, 然后 keepdim 再做除法，除法广播机制,
                 不过个人实测似乎说明不添加一范数的效果更好, 因而注释去掉了正规写法, 若有需要可以改回正规写法
        参考链接: https://zhuanlan.zhihu.com/p/170602502
        """
        # g = get_gradient(model, adv_images + images_grad, labels, criterion, device)
        # g_norm = torch.norm(g, p = 1, dim = (1, 2, 3), keepdim = True)
        # images_grad = miu * images_grad + (g / g_norm)

        images_grad = miu * images_grad + get_gradient(model, adv_images, labels, criterion, device)

        # 计算带有对抗扰动的图像，并将其剪裁到0和1之间; 赋值符号左右两侧出现同一个变量名的时候记得要用 detach,否则计算图可能会出错
        adv_noise = torch.sign(images_grad)
        adv_images = adv_images.detach() + alpha * adv_noise
        adv_images = torch.clamp(adv_images, min = minv, max = maxv)
    # 返回对抗扰动和标签列表
    return adv_noise, adv_images


def generate_ni_fgsm_noise(model, images, labels, criterion, device, epsilon=0.01, miu = 0.5, num_iter = 5, scale_copies = 5, minv = 0, maxv = 1):
    """
    传入模型model，图像images，生成带有 NI-FGSM 对抗噪声的图片
    :param model: 模型
    :param images: 图像
    :param labels: 图像标签
    :param device: 运算设备
    :param criterion: 损失函数
    :param epsilon: 噪声系数
    :param ni: NI 系数
    :param num_iter: 迭代次数
    :return: 对抗噪声, 带有 NI-FGSM 对抗噪声的图片
    """
    # 参考链接： https://zhuanlan.zhihu.com/p/497026313
    alpha = epsilon / num_iter
    # 获取图像的梯度
    adv_images = images.clone()
    images_grad = torch.zeros_like(images)
    for t in range(num_iter):
        g = torch.zeros_like(images)
        nes_images = adv_images + alpha * miu * images_grad
        for i in range(scale_copies):
            g += get_gradient(model, nes_images / (1 << i), labels, criterion, device)
        g /= scale_copies
        images_grad = miu * images_grad +  (g /  torch.norm(g, p = 1, dim = (1, 2, 3), keepdim = True))
        adv_images = adv_images + alpha * torch.sign(images_grad)
        adv_images = torch.clamp(adv_images, min = minv, max = maxv)
    # 返回对抗扰动和标签列表
    return adv_images - images, adv_images