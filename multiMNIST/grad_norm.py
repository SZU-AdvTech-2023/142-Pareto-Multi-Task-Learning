import argparse
import torch
import numpy as np
from model_lenet import RegressionModel, RegressionTrain
from torch.utils import data
import pickle
from model_resnet import MnistResNet, RegressionTrainResNet

def train_toy_example(dataset, base_model, niter, npref, init_weight, pref_idx, args):
    # # set the random seeds for reproducibility
    # np.random.seed(123)
    # torch.cuda.manual_seed_all(123)
    # torch.manual_seed(123)

    # define the sigmas, the number of tasks and the epsilons
    sigmas = [1.0, float(args.sigma)]
    print('Training with sigmas={}'.format(sigmas))
    n_tasks = len(sigmas)
    epsilons = np.random.normal(scale=3.5, size=(n_tasks, 100, 250)).astype(np.float32)

    # initialize the data loader
    if dataset == 'mnist':
        with open('data/multi_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

            # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('data/multi_fashion.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

            # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('data/multi_fashion_and_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))

    # initialize the model and use CUDA if available
    if base_model == 'lenet':
        model = RegressionTrain(RegressionModel(n_tasks), init_weight)
    if base_model == 'resnet18':
        model = RegressionTrainResNet(MnistResNet(n_tasks), init_weight)

    if torch.cuda.is_available():
        model.cuda()

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_iterations = int(args.n_iter)
    weights = []
    task_losses = []
    loss_ratios = []
    grad_norm_losses = []
    train_accs = []


    # run n_iter iterations of training
    for t in range(n_iterations):
        correct1_train = 0
        correct2_train = 0
        # get a single batch
        for (it, batch) in enumerate(train_loader):
            #  get the X and the targets values
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # evaluate each task loss L_i(t)
            task_loss = model(X, ts)  # this will do a forward pass in the model and will also evaluate the loss
            # compute the weighted loss w_i(t) * L_i(t)
            weighted_task_loss = torch.mul(model.weights, task_loss)

            # 计算准确率
            output1 = model.model(X).max(2, keepdim=True)[1][:, 0]
            output2 = model.model(X).max(2, keepdim=True)[1][:, 1]
            correct1_train += output1.eq(ts[:, 0].view_as(output1)).sum().item()
            correct2_train += output2.eq(ts[:, 1].view_as(output2)).sum().item()

            # initialize the initial loss L(0) if t=0
            if t == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                initial_task_loss = initial_task_loss.numpy()

            # get the total loss
            loss = torch.sum(weighted_task_loss)
            # clear the gradients
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            # This is equivalent to compute each \nabla_W L_i(t)
            loss.backward(retain_graph=True)

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            # print('Before turning to 0: {}'.format(model.weights.grad))
            model.weights.grad.data = model.weights.grad.data * 0.0
            # print('Turning to 0: {}'.format(model.weights.grad))

            # switch for each weighting algorithm:
            # --> grad norm
            if args.mode == 'grad_norm':

                # get layer of shared weights
                W = model.get_last_shared_layer()

                # get the gradient norms for each of the tasks
                # G^{(i)}_w(t)
                norms = []
                for i in range(len(task_loss)):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    # compute the norm
                    norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
                norms = torch.stack(norms)
                # print('G_w(t): {}'.format(norms))

                # compute the inverse training rate r_i(t)
                # \curl{L}_i
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # print('r_i(t): {}'.format(inverse_train_rate))

                # compute the mean norm \tilde{G}_w(t)
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                # print('tilde G_w(t): {}'.format(mean_norm))

                # compute the GradNorm loss
                # this term has to remain constant
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                # print('Constant term: {}'.format(constant_term))
                # this is the GradNorm loss itself
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]

            # do a step with the optimizer
            optimizer.step()
            '''
            print('')
            wait = input("PRESS ENTER TO CONTINUE.")
            print('')
            '''

        # renormalize
        normalize_coeff = n_tasks / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        train_acc = np.stack(
            [1.0 * correct1_train / len(train_loader.dataset), 1.0 * correct2_train / len(train_loader.dataset)])
        # record
        if torch.cuda.is_available():
            task_losses.append(task_loss.data.cpu().numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.cpu().numpy())
            grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
            # 记录准确率
            train_accs.append(train_acc)
        else:
            task_losses.append(task_loss.data.numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.numpy())
            grad_norm_losses.append(grad_norm_loss.data.numpy())
            train_accs.append(train_acc)

        if t == 0 or (t + 1) % 2 == 0:
            if torch.cuda.is_available():
                print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}, train_acc={}'.format(
                    t+1, args.n_iter, loss_ratios[-1], model.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(),
                    grad_norm_loss.data.cpu().numpy(), train_accs[-1]))
            else:
                print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}, train_acc={}'.format(
                    t+1, args.n_iter, loss_ratios[-1], model.weights.data.numpy(), task_loss.data.numpy(),
                    grad_norm_loss.data.numpy(), train_accs[-1]))

    task_losses = np.array(task_losses)
    weights = np.array(weights)

    torch.save(model.model.state_dict(), './GN_model/resnet18/mnist_resnet18/%s_%s_niter_%d_npref_%d_prefidx_%d.pickle' % (
    dataset, base_model, niter, npref, pref_idx))


def run(dataset = 'mnist',base_model = 'lenet', niter = 100, npref = 5):
    """
    run Grad Norm
    """
    parser = argparse.ArgumentParser(description='GradNorm')
    parser.add_argument('--n-iter', '-it', type=int, default=niter)
    parser.add_argument('--mode', '-m', choices=('grad_norm', 'equal_weight'), default='grad_norm')
    parser.add_argument('--alpha', '-a', type=float, default=0.12)
    parser.add_argument('--sigma', '-s', type=float, default=1.0)
    args = parser.parse_args()

    init_weight = np.array([0.5 , 0.5 ])

    for i in range(npref):

        pref_idx = i
        train_toy_example(dataset, base_model, niter, npref, init_weight, pref_idx, args)

if __name__ == '__main__':
    # run(dataset = 'mnist', base_model = 'lenet', niter = 100, npref = 5)
    # run(dataset = 'fashion', base_model = 'lenet', niter = 100, npref = 5)
    # run(dataset = 'fashion_and_mnist', base_model = 'lenet', niter = 100, npref = 5)
    #
    run(dataset = 'mnist', base_model = 'resnet18', niter = 20, npref = 5)
    # run(dataset = 'fashion', base_model = 'resnet18', niter = 20, npref = 5)
    # run(dataset='fashion_and_mnist', base_model='resnet18', niter=20, npref=5)
