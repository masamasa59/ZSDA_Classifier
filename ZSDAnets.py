import torch
from torch import nn
from torch.autograd import Variable


# INFERENCE NETWORK q(z|X_d)
class InferenceNetwork(nn.Module):
    """

    """

    def __init__(self, n_features, hidden_dim, z_dim, domain):
        super(InferenceNetwork, self).__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.domain = domain
        self.z_dim = z_dim
        self.eta_net = nn.Sequential(
                nn.Linear(self.n_features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()            
                )
        self.rho_net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,2*self.z_dim),
                nn.ReLU()
                )
        
    def forward(self,x):
        #x = [5domain,100sample, 256dim]
        eta = self.eta_net(x)#[5,100, 100]
        mean = torch.mean(eta,dim = 1)#100個ずつ平均をとる#[domain,hidden_dim]
        rho = self.rho_net(mean)
        mean, logvar = rho[:, :self.z_dim], rho[:, self.z_dim:]

        return mean, logvar


# Observation Decoder p(x|z)
class ObservationClassifier(nn.Module):
    #f = softmax(h(x_d)^Tg(z_d))
    def __init__(self, n_features, n_c, hidden_dim, J_dim, z_dim, domain):
        super(ObservationClassifier, self).__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.n_c = n_c #the number of class
        self.J_dim = J_dim #innner_product_dim
        self.z_dim = z_dim #latent_variable
        self.domain = domain #number domain
        self.h_func = nn.Sequential(
                      nn.Linear(self.n_features, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      #nn.Linear(self.hidden_dim, self.n_c),
                      #nn.ReLU()
                    )
        self.g_func_1 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_2 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_3 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_4 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_5 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_6 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_7 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        
        self.g_func_8 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_9 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim),
                      nn.Tanh()
                    )
        self.g_func_10 = nn.Sequential(
                      nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.J_dim*self.n_c),
                      nn.Tanh()
                    )
    def forward(self,x,z):
        h = self.h_func(x) #[5dom, 100sample, J_dim]
        """
        g1 = self.g_func_1(z) #[5dom, J_dim]classiffer1
        g2 = self.g_func_2(z) #[5dom, J_dim]classiffer2
        g3 = self.g_func_3(z) #[5dom, J_dim]classiffer3
        g4 = self.g_func_4(z) #[5dom, J_dim]classiffer4
        g5 = self.g_func_5(z) #[5dom, J_dim]classiffer5
        g6 = self.g_func_6(z) #[5dom, J_dim]classiffer6
        g7 = self.g_func_7(z) #[5dom, J_dim]classiffer7
        g8 = self.g_func_8(z) #[5dom, J_dim]classiffer8
        g9 = self.g_func_9(z) #[5dom, J_dim]classiffer9
        g10 = self.g_func_10(z) #[5dom, J_dim]classiffer10
        """
        g = self.g_func_10(z)
        g = g.view(x.size(0), self.J_dim, self.n_c)#[5dom, J_dim, n_c]
        #g = torch.cat((g1, g2, g3, g4, g5, g6, g7, g8, g9, g10), 0).view(x.size(0), self.J_dim, self.n_c)
        f = nn.functional.log_softmax(torch.matmul(h,g),dim=1) #[domain:5,samples:100, class:10]
        #f = nn.functional.log_softmax(h,dim=1)
        
        return f.view(-1,self.n_c) #[samples500 ,n_c]


