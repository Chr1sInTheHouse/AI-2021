import numpy as np
import dataset
import torch

seed = 46
np.random.seed(seed)
torch.manual_seed(seed)

class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, alpha, nonLinear=False):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        """

        d = len(dataset.features[
                    0]) * 2  # size for A, b matrices: num of features for articles(6) + num of features for users(6) = 12
        self.A = np.array([np.identity(d)] * dataset.n_arms)
        self.b = np.zeros((dataset.n_arms, d, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB (α=" + str(self.alpha) + ")"
        self.theta = np.abs(np.random.randn(1, d))
        self.theta /= np.linalg.norm(self.theta, ord=2)

        self.cum_regret = []
        self.nonLinear = nonLinear

    def choose_arm(self, t, user, pool_idx):
        """
         Returns the best arm's index relative to the pool
         Parameters
         ----------
         t : number
             number of trial
         user : array
             user features
         pool_idx : array of indexes
             pool indexes for article identification
         """

        A = self.A[pool_idx]  # (23, 12, 6)
        b = self.b[pool_idx]  # (23, 12, 1)
        user = np.array([user] * len(pool_idx))  # (23, 6)

        A = np.linalg.inv(A)
        x = np.hstack((user, dataset.features[
            pool_idx]))  # (23, 12) The vector x summarizes information of both the user u and arm a

        x = x.reshape((len(pool_idx), 12, 1))  # (23, 12, 1)

        reward = x.reshape(len(pool_idx), 12) @ self.theta.T
        if self.nonLinear:
            reward = 3*np.square(reward)
        best_arm = np.argmax(reward)
        best_reward = np.max(reward)

        theta = A @ b  # (23, 12, 1)

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A @ x
        )
        arm = np.argmax(p)
        self.cum_regret.append(best_reward - reward[arm])
        return arm, reward[arm]


    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]  # displayed article's index

        x = np.hstack((user, dataset.features[a]))
        x = x.reshape((12, 1))

        self.A[a] = self.A[a] + x @ np.transpose(x)
        self.b[a] += reward * x


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """
    def __init__(self, epsilon, nonLinear=False):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """

        self.e = round(epsilon, 1)  # epsilon parameter for Egreedy
        self.algorithm = "Egreedy (ε=" + str(self.e) + ")"
        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.zeros(dataset.n_arms)  # number of times each arm was chosen
        self.theta = np.abs(np.random.randn(1, 19))
        self.theta /= np.linalg.norm(self.theta, ord=2)

        self.cum_regret = []
        self.nonLinear = nonLinear

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        reward = self.theta.T
        if self.nonLinear:
            reward = 3*np.square(reward)
        best_arm = np.argmax(reward)
        best_reward = np.max(reward)

        p = np.random.rand()
        if p > self.e:
            arm = np.argmax(self.q[pool_idx])
        else:
            arm = np.random.randint(low=0, high=len(pool_idx)-1)
        self.cum_regret.append(best_reward - reward[arm])
        return arm, reward[arm]

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not            
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]

import torch
from  torch import linalg as LA
import torch.nn as nn
import torch.nn.functional as F
import math

class Net(nn.Module):
    def __init__(self, d, m, l):
        super(Net, self).__init__()

        self.l = l
        self.m = m
        self.d = d
        self.fc1 = nn.Linear(d, m, False).cuda()
        self.fc2 = nn.Linear(m, 1, False).cuda()

    def forward(self, x):

        for i in range(self.l-1):
            out = self.fc1(x.float())
            out = F.relu(out.float())
            x = out
        x = self.fc2(x.float())
        x = math.sqrt(self.m) * x

        return x

class NeuralUCB:
    """
    NeuralUCB algorithm implementation
    """

    def __init__(self, alpha, nonLinear=False):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        """

        d = len(dataset.features[
                    0]) * 2  # size for A, b matrices: num of features for articles(6) + num of features for users(6) = 12
        # A = np.array([np.identity(d)] * dataset.n_arms)
        # self.b = np.zeros((dataset.n_arms, d, 1))
        self.alpha = alpha
        self.algorithm = "NeuralUCB (α=" + str(self.alpha) + ")"
        self.D = d
        self.M = 20
        self.L = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.theta_n = self.init_theta(D, M, L).to(self.device)
        self.theta_n = torch.rand((1, self.D*self.M+self.M)).to(self.device)
        self.theta_0 = self.theta_n.clone()
        self.theta_len =  self.theta_n.shape[1]
        A = np.array([np.identity(self.theta_len) * 1e-2] * dataset.n_arms)
        self.A = torch.tensor(A).to(self.device)

        self.init = True
        self.grad = torch.empty((1, 1)).to(self.device)
        self.used_grad = torch.empty((1, 1)).to(self.device)
        self.X = torch.empty((1, 1)).to(self.device)
        self.reward = torch.empty((1, 1)).to(self.device)
        
        self.theta = np.abs(np.random.randn(1, self.D))
        self.theta /= np.linalg.norm(self.theta, ord=2)

        self.cum_regret = []
        self.nonLinear = nonLinear


    def choose_arm(self, t, user, pool_idx):
        """
         Returns the best arm's index relative to the pool
         Parameters
         ----------
         t : number
             number of trial
         user : array
             user features
         pool_idx : array of indexes
             pool indexes for article identification
         """

        user = np.array([user] * len(pool_idx))  # (23, 6)

        A = self.A[pool_idx]  # (23, 12, 6)
        contexts = np.hstack((user, dataset.features[
            pool_idx]))  # (23, 12) The vector x summarizes information of both the user u and arm a

        reward = contexts @ self.theta.T
        # print(reward)
        if self.nonLinear:
            reward = 3*np.square(reward)
        best_arm = np.argmax(reward)
        best_reward = np.max(reward)
        # contexts = contexts.reshape((len(pool_idx), 12))  # (23, 12)

        u_t = np.zeros(len(pool_idx))
        f = []
        g = []
        x = []

        for k in range(len(pool_idx)):
            temp = torch.tensor(contexts[k]).to(self.device)
            x.append(torch.autograd.Variable(temp, requires_grad=True))

        for k in range(len(pool_idx)):
            model = Net(self.D, self.M, self.L)

            # print(model.state_dict())
            model.state_dict()['fc1.weight'][:] = torch.narrow(self.theta_n, 1, 0, self.D*self.M).reshape(self.M, self.D)
            model.state_dict()['fc2.weight'][:] = torch.narrow(self.theta_n, 1, self.D*self.M, self.M).reshape(1, self.M)

            out = model.forward(x[k])
            model.zero_grad()
            f.append(out)
            out.backward()
            g_temp = model.fc1.weight.grad.flatten()
            g_temp = torch.cat((g_temp, model.fc2.weight.grad.flatten())).reshape(self.D*self.M+self.M, 1)
            g.append(g_temp)
        for k in range(len(pool_idx)):
            u_t[k] = f[k].float() + self.alpha * torch.sqrt((torch.mm(torch.mm(g[k].t().float(), torch.inverse(A[k].float())), g[k].float())/self.M))
        
        arm = np.argmax(u_t)

        if(t==1):
            self.grad = g[arm]
        else:
            self.grad = torch.cat((self.grad, g[arm]), dim=1)

        self.t = t
        self.cum_regret.append(best_reward - reward[arm])
        # print(f't:{t} arm: {arm} good: {best_arm}')
        return arm, reward[arm]


    def update(self, displayed, r_now, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        r_now : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]  # displayed article's index

        x = np.hstack((user, dataset.features[a]))
        x = x.reshape((1, 12))
        x = torch.tensor(x).to(self.device)
        grad_now = self.grad[:,-1].reshape(self.theta_len, 1)
        r_now = torch.tensor(r_now).to(self.device)
        r_now = r_now.reshape((1, 1))

        if(self.init):
            self.init = False
            self.X = x
            self.used_grad = grad_now
            self.reward = r_now
        else:
            self.X = torch.cat((self.X, x), dim=0)
            self.used_grad = torch.cat((self.used_grad, grad_now), dim=1)
            self.reward = torch.cat((self.reward, r_now), dim=0)
        
        self.A[a] = self.A[a] + torch.mm(grad_now, grad_now.t()) / self.M
        self.theta_n = self.TrainNN(1e-2, 1e-5, self.t if self.t<50 else 50, self.X, self.reward, self.theta_0, self.used_grad)


    def TrainNN(self, lda, eta, U, x, r, theta_0, g):
        theta_now = theta_0.clone()

        for i in range(U):
            theta_grad, ll = self.LOSS(x, r, theta_now, lda, theta_0, g)
            theta_now = theta_now - eta*theta_grad
        
        return theta_now

    def LOSS(self, x, r, theta_now, lda, theta_0, g):
        torch.autograd.Variable(x, requires_grad=True)
        model = Net(self.D, self.M, self.L)
        model.state_dict()['fc1.weight'][:] = torch.narrow(theta_now, 1, 0, self.D*self.M).reshape(self.M, self.D)
        model.state_dict()['fc2.weight'][:] = torch.narrow(theta_now, 1, self.D*self.M, self.M).reshape(1, self.M)

        fx = model.forward(x)
        model.zero_grad()

        loss = nn.MSELoss(reduction='mean')
        ll = loss(fx.double(), r.double())
        norm = torch.norm(theta_now-theta_0) * self.M * lda
        ll += norm
        ll.backward()

        theta_grad = model.fc1.weight.grad.flatten()
        theta_grad = torch.cat((theta_grad, model.fc2.weight.grad.flatten())).reshape(1, self.theta_len)

        return theta_grad, ll

        
