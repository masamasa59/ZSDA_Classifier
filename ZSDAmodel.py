import os
import sys
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, init
from math import log, pi
from ZSDAnets import InferenceNetwork, ObservationClassifier
# Model
class ZSDA(nn.Module):
    def __init__(self, batch_size=512, sample_size=1000, n_features=256,
                z_dim=4, hidden_dim=100, n_c=10, J_dim=10, print_vars=False,n_domain=5):
        """
        :param batch_size: 
        :param sample_size: 
        :param n_features: 
        :param z_dim:  
        :param hidden_dim: 
        :n_c:
        :J_dim:
        :param print_vars: 
        :n_domain:
        """
        super(ZSDA, self).__init__()
        # data shape
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.n_c = n_c
        self.domain = n_domain
        # latent
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.J_dim = J_dim
        
        # modules
        z_args = ( self.n_features, self.hidden_dim, self.z_dim, self.domain )
        # inference networks
        self.inference_network = InferenceNetwork(*z_args)

        # observation decoder
        observation_args = ( self.n_features, self.n_c, self.hidden_dim, self.J_dim, self.z_dim, self.domain)
        
        self.observation_classifier = ObservationClassifier(*observation_args)

        # initialize weights
        #self.apply(self.weights_init)

        # print variables for sanity check and debugging
        if print_vars:
            for i, pair in enumerate(self.named_parameters()):
                name, param = pair
                print("{} --> {}, {}".format(i + 1, name, param.size()))
            print()

    def forward(self, x, y):
        # statistic network
        z_mean, z_logvar = self.inference_network(x)#[5domain,100samples,256dim]
        z = self.reparameterize_gaussian(z_mean, z_logvar)#[5domain,10hidden_dim]

        # observation decoder
        #zs = torch.cat(z, dim=1)
        y_pred = self.observation_classifier(x,z)

        outputs = (
            (z_mean, z_logvar),
            (y, y_pred)
        )

        return outputs

    def loss(self, outputs, criterion, weight):

        z_outputs, y_outputs = outputs
        y, y_pred = y_outputs
        # 1. Reconstruction loss
        recon_loss = criterion(y_pred,y)
        #recon_loss /= (self.batch_size * self.sample_size)
        # 2. KL Divergence terms
        kl = 0

        # a) Context divergence
        z_mean, z_logvar = z_outputs
        kl_z = kl_diagnormal_stdnormal(z_mean, z_logvar)
        kl += kl_z
        
        _, predicted = torch.max(y_pred,1)#予測ラベル
        correct = (predicted == y).sum().item()
        loss_acc = (float(correct) / y.size(0))*100
     
        # Variational lower bound and weighted loss
        vlb = - recon_loss - kl
        loss =  (weight*recon_loss + kl/weight )#- (kl/weight))
        #print(kl.data)
        #print(recon_loss.data)
        return loss, vlb, loss_acc, z_outputs

    def step(self, batch, alpha, criterion, optimizer, clip_gradients=True):
        assert self.training is True
        inputs, y = batch
        inputs, y = Variable(inputs.cuda()),Variable(y.cuda())
        optimizer.zero_grad()
        outputs = self.forward(inputs,y)
        loss, vlb, loss_acc, z_outputs = self.loss(outputs, criterion, 300)
        # perform gradient update
        loss.backward()
        #if clip_gradients:
        #    for param in self.parameters():
        #        if param.grad is not None:
        #            param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
        optimizer.step()

        # output variational lower bound
        return loss, vlb.data, loss_acc, z_outputs

    def save(self, optimizer, path):
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, path)

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()).cuda())
        return mean + std * eps

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            init.xavier_normal(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    a = mean**2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)


def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    return 0.5 * torch.sum(a + b + c + d)