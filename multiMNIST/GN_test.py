import torch
import pickle
import numpy as np
from model_lenet import RegressionModel, RegressionTrain
from model_resnet import MnistResNet, RegressionTrainResNet


def write_data(data, dataset, base_model):
    data_path = "4/" + dataset + "_" + base_model + "_" + "test_data.txt"
    with open(data_path, "a+") as f:
        f.write(data + '\n')


# 验证过程
def test(dataset, base_model):
    n_tasks = 2
    init_weight = np.array([0.5, 0.5])
    # load dataset

    # MultiMNIST: multi_mnist.pickle
    # Loading saved models
    if dataset == 'mnist':
        with open('data/multi_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)
        if base_model == 'lenet':
            with open('GN_model/lenet/mnist_lenet/mnist_lenet_niter_100_npref_5_prefidx_4.pickle', 'rb') as f:
                state_dict = torch.load(f)
                model = RegressionTrain(RegressionModel(n_tasks), init_weight)
                model.model.load_state_dict(state_dict)
        if base_model == 'resnet18':
            with open('GN_model/resnet18/mnist_resnet18/mnist_resnet18_niter_20_npref_5_prefidx_4.pickle', 'rb') as f:
                state_dict = torch.load(f)
                model = RegressionTrainResNet(MnistResNet(n_tasks), init_weight)
                model.model.load_state_dict(state_dict)

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('data/multi_fashion.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)
        if base_model == 'lenet':
            with open('GN_model/lenet/fashion_lenet/fashion_lenet_niter_100_npref_5_prefidx_4.pickle', 'rb') as f:
                state_dict = torch.load(f)
                model = RegressionTrain(RegressionModel(n_tasks), init_weight)
                model.model.load_state_dict(state_dict)
        if base_model == 'resnet18':
            with open('GN_model/resnet18/fashion_resnet18/fashion_resnet18_niter_20_npref_5_prefidx_4.pickle', 'rb') as f:
                state_dict = torch.load(f)
                model = RegressionTrainResNet(MnistResNet(n_tasks), init_weight)
                model.model.load_state_dict(state_dict)

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('data/multi_fashion_and_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)
        if base_model == 'lenet':
            with open('GN_model/lenet/fashion_and_mnist_lenet/fashion_and_mnist_lenet_niter_100_npref_5_prefidx_4.pickle', 'rb') as f:
                state_dict = torch.load(f)
                model = RegressionTrain(RegressionModel(n_tasks), init_weight)
                model.model.load_state_dict(state_dict)
        if base_model == 'resnet18':
            with open('GN_model/resnet18/fashion_and_mnist_resnet18/fashion_and_mnist_resnet18_niter_20_npref_5_prefidx_4.pickle', 'rb') as f:
                state_dict = torch.load(f)
                model = RegressionTrainResNet(MnistResNet(n_tasks), init_weight)
                model.model.load_state_dict(state_dict)


    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()

    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = 256
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    # 判断CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将模型移动到CUDA设备上
    model = model.to(device)

    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)

    weights = []
    task_test_losses = []
    test_accs = []

    with torch.no_grad():

        total_test_loss = []
        test_acc = []

        correct1_test = 0
        correct2_test = 0

        for (it, batch) in enumerate(test_loader):

            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            valid_test_loss = model(X, ts)
            total_test_loss.append(valid_test_loss)
            output1 = model.model(X).max(2, keepdim=True)[1][:, 0]
            output2 = model.model(X).max(2, keepdim=True)[1][:, 1]
            correct1_test += output1.eq(ts[:, 0].view_as(output1)).sum().item()
            correct2_test += output2.eq(ts[:, 1].view_as(output2)).sum().item()

        test_acc = np.stack(
            [1.0 * correct1_test / len(test_loader.dataset), 1.0 * correct2_test / len(test_loader.dataset)])

        total_test_loss = torch.stack(total_test_loss)
        average_test_loss = torch.mean(total_test_loss, dim=0)

    # record and print
    if torch.cuda.is_available():
        task_test_losses.append(average_test_loss.data.cpu().numpy())
        test_accs.append(test_acc)

        data = 'test_loss={}, test_acc={}'.format(task_test_losses[-1], test_accs[-1])
        write_data(data, dataset, base_model)
        print(data)


# test(dataset = 'mnist', base_model = 'lenet')
test(dataset = 'fashion', base_model = 'lenet')
test(dataset = 'fashion_and_mnist', base_model = 'lenet')
test(dataset = 'mnist', base_model = 'resnet18')
test(dataset = 'fashion', base_model = 'resnet18')
test(dataset = 'fashion_and_mnist', base_model = 'resnet18')
