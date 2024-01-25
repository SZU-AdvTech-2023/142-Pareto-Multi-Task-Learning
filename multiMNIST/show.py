# import matplotlib.pyplot as plt
# import re
# import ast
#
# # 从txt文件中读取数据
# data = []
# with open('fashion_and_mnist_resnet18_niter_20_npref_5_prefidx_4_train_result.txt', 'r') as file:
#     lines = file.readlines()[5:]  # 跳过前五行
#     for line in lines:
#         line = line.strip()
#         if line:
#             line_data = line.split(': ')
#             iteration = int(line_data[0].split('/')[0])
#             line_data = line.split(', ')
#             weights = (line_data[0].split('=')[1])
#             weights = re.findall(r"\d+\.\d+", weights)
#             train_loss = line_data[1].split('=')[1].strip('[]')
#             if 'e' in train_loss:
#                 # 处理包含科学计数法的数据
#                 train_loss = [float(x) for x in train_loss.split()]
#             else:
#                 # 处理普通浮点数表示的数据
#                 train_loss = [float(x) for x in train_loss.split()]
#             train_acc = (line_data[2].split('=')[1])
#             train_acc = re.findall(r"\d+(?:\.\d+)?", train_acc)
#             data.append((iteration, weights, train_loss, train_acc))
#
# # 提取数据并绘图
# iterations = [d[0] for d in data]
# train_loss_1 = [float(d[2][0]) for d in data]
# train_loss_2 = [float(d[2][1]) for d in data]
# train_acc_1 = [float(d[3][0]) for d in data]
# train_acc_2 = [float(d[3][1]) for d in data]
#
# plt.figure(figsize=(10, 5))
# # plt.subplot(1, 2, 1)
#
# plt.plot(iterations, train_loss_1, label='Train Loss 1')
# plt.plot(iterations, train_loss_2, label='Train Loss 2')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Training Loss 4')
# plt.legend()
# plt.show()
# plt.clf()
#
# # plt.subplot(1, 2, 2)
# plt.plot(iterations, train_acc_1, label='Train Acc 1')
# plt.plot(iterations, train_acc_2, label='Train Acc 2')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy 4')
# plt.legend()
#
#
# plt.tight_layout()
# plt.show()
# plt.clf()

import matplotlib.pyplot as plt

# ParetoMTL数据
# MNIST数据
PMTL_mnist_resnet18_top_left = [0.94965, 0.9549, 0.95575, 0.95745, 0.9567]
PMTL_mnist_resnet18_bottom_right = [0.94755, 0.9466, 0.94765, 0.94175, 0.92885]
PMTL_mnist_lenet_top_left = [0.8928, 0.8915, 0.8827, 0.91065, 0.91895]
PMTL_mnist_lenet_bottom_right = [0.88935, 0.88825, 0.84895, 0.86765, 0.8183]

# FASHION数据
PMTL_fashion_resnet18_top_left = [0.86275, 0.86055, 0.86855, 0.87135, 0.87515]
PMTL_fashion_resnet18_bottom_right = [0.8655, 0.8644, 0.8598, 0.85065, 0.85145]
PMTL_fashion_lenet_top_left = [0.6657, 0.76615, 0.7667, 0.82045, 0.8276]
PMTL_fashion_lenet_bottom_right = [0.821, 0.8154, 0.78365, 0.76615, 0.6717]

# FASHION AND MNIST数据
PMTL_fashion_mnist_resnet18_top_left = [0.94315, 0.94785, 0.9633, 0.97505, 0.9809]
PMTL_fashion_mnist_resnet18_bottom_right = [0.88985, 0.8868, 0.8873, 0.88235, 0.854]
PMTL_fashion_mnist_lenet_top_left = [0.7336, 0.851, 0.9025, 0.9288, 0.9449]
PMTL_fashion_mnist_lenet_bottom_right = [0.85255, 0.85635, 0.84385, 0.81825, 0.79055]


# GradNorm数据
# MNIST数据
gradnorm_mnist_resnet18_top_left = [0.9287, 0.9275, 0.92665, 0.92855, 0.93195]
gradnorm_mnist_resnet18_bottom_right = [0.90885, 0.91375, 0.91275, 0.90325, 0.9063]
gradnorm_mnist_lenet_top_left = [0.91265, 0.91535, 0.9176, 0.91535, 0.91405]
gradnorm_mnist_lenet_bottom_right = [0.8919, 0.8988, 0.8929, 0.8986, 0.89605]

# FASHION数据
gradnorm_fashion_resnet18_top_left = [0.82195, 0.82145, 0.8207, 0.8226, 0.8227]
gradnorm_fashion_resnet18_bottom_right = [0.8087, 0.8086, 0.81465, 0.8079, 0.80985]
gradnorm_fashion_lenet_top_left = [0.809, 0.8164, 0.81, 0.8245, 0.81615]
gradnorm_fashion_lenet_bottom_right = [0.8034, 0.8103, 0.8051, 0.8097, 0.80745]

# FASHION AND MNIST数据
gradnorm_fashion_mnist_resnet18_top_left = [0.93265, 0.9288, 0.9267, 0.93415, 0.9289]
gradnorm_fashion_mnist_resnet18_bottom_right = [0.8377, 0.8365, 0.83565, 0.83125, 0.83315]
gradnorm_fashion_mnist_lenet_top_left = [0.9274, 0.929, 0.92685, 0.93095, 0.92825]
gradnorm_fashion_mnist_lenet_bottom_right = [0.83125, 0.8302, 0.8327, 0.8313, 0.8357]
# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# 绘制MNIST数据散点图
axs[0, 0].scatter(PMTL_mnist_lenet_top_left, PMTL_mnist_lenet_bottom_right, label='ParetoMTL')
axs[0, 0].scatter(gradnorm_mnist_lenet_top_left, gradnorm_mnist_lenet_bottom_right, label='GradNorm')
axs[0, 0].set_xlabel('Task 1: Top-Left')
axs[0, 0].set_ylabel('Task 2: Bottom-Right')
axs[0, 0].set_title('MultiMNIST - LeNet')
axs[0, 0].legend()

axs[1, 0].scatter(PMTL_mnist_resnet18_top_left, PMTL_mnist_resnet18_bottom_right, label='ParetoMTL')
axs[1, 0].scatter(gradnorm_mnist_resnet18_top_left, gradnorm_mnist_resnet18_bottom_right, label='GradNorm')
axs[1, 0].set_xlabel('Task 1: Top-Left')
axs[1, 0].set_ylabel('Task 2: Bottom-Right')
axs[1, 0].set_title('MultiMNIST - ResNet18')
axs[1, 0].legend()

# 绘制FASHION数据散点图
axs[0, 1].scatter(PMTL_fashion_lenet_top_left, PMTL_fashion_lenet_bottom_right, label='ParetoMTL')
axs[0, 1].scatter(gradnorm_fashion_lenet_top_left, gradnorm_fashion_lenet_bottom_right, label='GradNorm')
axs[0, 1].set_xlabel('Task 1: Top-Left')
axs[0, 1].set_ylabel('Task 2: Bottom-Right')
axs[0, 1].set_title('MultiFashionMNIST - LeNet')
axs[0, 1].legend()

axs[1, 1].scatter(PMTL_fashion_resnet18_top_left, PMTL_fashion_resnet18_bottom_right, label='ParetoMTL')
axs[1, 1].scatter(gradnorm_fashion_resnet18_top_left, gradnorm_fashion_resnet18_bottom_right, label='GradNorm')
axs[1, 1].set_xlabel('Task 1: Top-Left')
axs[1, 1].set_ylabel('Task 2: Bottom-Right')
axs[1, 1].set_title('MultiFashionMNIST - ResNet18')
axs[1, 1].legend()

# 绘制FASHION AND MNIST数据散点图
axs[0, 2].scatter(PMTL_fashion_mnist_lenet_top_left, PMTL_fashion_mnist_lenet_bottom_right, label='ParetoMTL')
axs[0, 2].scatter(gradnorm_fashion_mnist_lenet_top_left, gradnorm_fashion_mnist_lenet_bottom_right, label='GradNorm')
axs[0, 2].set_xlabel('Task 1: Top-Left')
axs[0, 2].set_ylabel('Task 2: Bottom-Right')
axs[0, 2].set_title('Multi-(Fashion+MNIST) - LeNet')
axs[0, 2].legend()

axs[1, 2].scatter(PMTL_fashion_mnist_resnet18_top_left, PMTL_fashion_mnist_resnet18_bottom_right, label='ParetoMTL')
axs[1, 2].scatter(gradnorm_fashion_mnist_resnet18_top_left, gradnorm_fashion_mnist_resnet18_bottom_right, label='GradNorm')
axs[1, 2].set_xlabel('Task 1: Top-Left')
axs[1, 2].set_ylabel('Task 2: Bottom-Right')
axs[1, 2].set_title('Multi-(Fashion+MNIST) - ResNet18')
axs[1, 2].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
