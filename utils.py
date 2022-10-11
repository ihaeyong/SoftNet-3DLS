#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from pyhessian import hessian
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_losses(epochs, dense_losses, hardnet_losses, softnet_losses):
    # plot losses 
    plt.plot(range(epochs), dense_losses, label='DenseNet')
    plt.plot(range(epochs), hardnet_losses, label='HardNet')
    plt.plot(range(epochs), softnet_losses, label='SoftNet')
    plt.legend(fontsize=18)
    # Set the font size for x tick labels
    plt.rc('xtick', labelsize=20)
    # Set the font size for y tick labels
    plt.rc('ytick', labelsize=20)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.savefig('./plots/loss_curves.pdf', dpi=300, format='pdf')
    plt.close()


def pca_input_plot(x):
    x = StandardScaler().fit_transform(x) # x객체에 x를 표준화한 데이터를 저장

    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    pd.DataFrame(x, columns=features).head()

    pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
    printcipalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])

    # 주성분으로 이루어진 데이터 프레임 구성
    print(principalDf.head())
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

    # scatter datapoints of 
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component1']
                   , finalDf.loc[indicesToKeep, 'principal component2']
                   , c = color
                   , s = 50)
        ax.legend(targets)
        ax.grid()

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def get_2d_params(model_orig,  model_perb, direction, alpha1, alpha2):
    for m_orig, m_perb, d1, d2 in zip(model_orig.parameters(), model_perb.parameters(), direction[0], direction[1]):
        m_perb.data = m_orig.data + alpha1 * d1 + alpha2 * d2
    return model_perb

def get_2d_named_params(model_orig,  model_perb, direction, alpha1, alpha2):

    idx = 0
    for (_, m_perb), (name, m_orig) in zip(model_perb.named_parameters(), model_orig.named_parameters()):
        if 'w_m' in name:
            continue

        d1 = direction[0][idx]
        d2 = direction[1][idx]
        m_perb.data = m_orig.data + alpha1 * d1 + alpha2 * d2
        idx += 1
    return model_perb


def get_1d_named_params(model_orig,  model_perb, direction, alpha):

    idx = 0
    for (_, m_perb), (name, m_orig) in zip(model_perb.named_parameters(), model_orig.named_parameters()):
        if 'w_m' in name:
            continue

        d = direction[idx]
        m_perb.data = m_orig.data + alpha * d
        idx += 1

    return model_perb


def loss_landscape(model, criterion, X_test, y_test, net_type, mask=None):

    # create the hessian computation module
    hessian_comp = hessian(model, criterion, data=(X_test, y_test), cuda=False, mask=mask)

    # Now let's compute the top 2 eigenavlues and eigenvectors of the Hessian
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    print("The top two eigenvalues of this model are: %.4f %.4f"% (top_eigenvalues[-1],top_eigenvalues[-2]))

    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams_x = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    lams_y = np.linspace(-0.5, 0.5, 21).astype(np.float32)

    loss_list = []
    loss_list2d = []

    # create a copy of the model
    model_perb1 = deepcopy(model)
    model_perb1.eval()
    model_perb1 = model_perb1.cuda()

    model_perb2 = deepcopy(model)
    model_perb2.eval()
    model_perb2 = model_perb2.cuda()

    # 2d-plot
    if False:
        for lam in lams:
            model_perb = get_params(model, model_perb, top_eigenvector[0], lam)
            loss_list.append(criterion(model_perb(X_test), y_test).item())

        plt.plot(lams, loss_list)
        plt.ylabel('Lossname')
        plt.xlabel('Perturbation')
        plt.title('Loss landscape perturbed based on top Hessian eigenvector')

        plt.savefig('./plots/{}_loss_func_surface.pdf'.format(net_type),
                    dpi=300, format='pdf', bbox_inches='tight')
        plt.close()

    # 3d-plot
    else:
        for lam_x in lams_x:
            for lam_y in lams_y:

                if mask is not None:
                    model_perb1 = get_1d_named_params(model, model_perb1, top_eigenvector[0], lam_x)
                    model_perb2 = get_1d_named_params(model_perb1, model_perb2, top_eigenvector[1], lam_y)
                    loss = criterion(model_perb2.forward(X_test, mask), y_test).item()
                    loss_list.append(loss)
                else:
                    model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam_x)
                    model_perb2 = get_params(model_perb1, model_perb2, top_eigenvector[1], lam_y)
                    loss = criterion(model_perb2(X_test), y_test).item()
                    loss_list.append(loss)

                loss_list2d.append((lam_x, lam_y, loss))

        #plt.savefig('./plots/{}_loss_func_surface.pdf'.format(net_type),
        #            dpi=300, format='pdf', bbox_inches='tight')
        #plt.close()
        X, Y = np.meshgrid(lams_x, lams_y)
        Z = np.array(loss_list).reshape(21, 21)

        fig = plt.figure()
        fig.set_size_inches(10, 10)

        ax = plt.axes(projection='3d')
        ax.set_zlim([0, 10])

        ax.set_xlabel(r'$\epsilon_1$', fontsize=20)
        ax.set_ylabel(r'$\epsilon_2$', fontsize=20)
        ax.set_zlabel('Loss', fontsize=14)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        #ax.set_title('Loss Landscape with N({:.3f}, {:.6f}) of Z, depth:{:.6f}'.format(Z.mean(), Z.var(), Z.max()-Z.min()))

        plt.title('Loss landscape perturbed based on two Hessian eigenvector')
        plt.savefig('./plots/{}_loss_func_surface_3D.pdf'.format(net_type),
                    dpi=300, format='pdf')
        plt.close()
        print("{}s min: {}".format(net_type, Z.min()))

    return np.array(loss_list2d)


def plot_trisurf(loss_dense, loss_hardnet, loss_softnet, epoch=100):

    fig = plt.figure()
    fig.set_size_inches(10, 10)

    ax = plt.axes(projection='3d')
    ax.set_zlim([0, 10])

    ax.set_xlabel(r'$\epsilon_1$', fontsize=18)
    ax.set_ylabel(r'$\epsilon_2$', fontsize=18)
    ax.set_zlabel('Loss', fontsize=18)

    ax.plot_trisurf(loss_dense[:,0],
                    loss_dense[:,1],
                    loss_dense[:,2], alpha=0.7, cmap='viridis', label='DensNet')

    ax.plot_trisurf(loss_hardnet[:,0],
                    loss_hardnet[:,1],
                    loss_hardnet[:,2], alpha=0.7, cmap='hot', label='HardNet')

    ax.plot_trisurf(loss_softnet[:,0],
                    loss_softnet[:,1],
                    loss_softnet[:,2], alpha=0.7, cmap='coolwarm', label='SoftNet')

    z_max = max(max(loss_dense[:,2]),max(loss_hardnet[:,2]), max(loss_softnet[:,2]))
    ax.set_zlim(0, 15)
    ax.view_init(elev=30, azim=45)

    #ax.legend()

    #plt.legend(fontsize=24)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    #plt.title('Loss landscape perturbed based on two Hessian eigenvector')
    file_name = './plots/loss_func_surface_3D_epoch{}.pdf'.format(epoch) 
    plt.savefig(file_name, dpi=300, format='pdf')
    plt.close()

    print(file_name)

def getData(name='cifar10', train_bs=128, test_bs=1000):
    """
    Get the dataloader
    """
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)
    if name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)

    return train_loader, test_loader


def test(model, test_loader, cuda=True):
    """
    Get the test performance
    """
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num
