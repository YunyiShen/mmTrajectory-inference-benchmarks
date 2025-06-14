import torch.nn as nn
import torch
import torchsde 

class LotkaVolterra(nn.Module):

    def __init__(self, alpha = 1., beta = .4, gamma = .4, delta = .1, sigma = .02, propvola = True):
        super(LotkaVolterra, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self.delta = nn.Parameter(torch.log(torch.tensor(delta)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.propvola = propvola
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        alpha = self.preprocess(self.alpha) # things needs to be positive
        beta = self.preprocess(self.beta)
        gamma = self.preprocess(self.gamma)
        delta = self.preprocess(self.delta)
        y = torch.relu(y) + 1e-10 # avoid going to 0
        dxdt =  alpha * y[:,0] - beta * y[:,0] * y[:,1]
        dydt = delta * y[:,0] * y[:,1] - gamma * y[:,1]
        return torch.stack([dxdt, dydt], dim = 1)
    
    def g(self, t, y):
        sigma = self.preprocess(self.sigma)
        if self.propvola:
            return sigma * torch.relu(y)
        else:
            return sigma + torch.zeros_like(y)
    

class lorenz(nn.Module):
    def __init__(self, alpha = 10, beta = 28, rho = 8/3, sigma = 1):
        super(lorenz, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.rho = nn.Parameter(torch.log(torch.tensor(rho)))
        
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        
        beta = self.preprocess(self.beta)
        alpha = self.preprocess(self.alpha)
        rho = self.preprocess(self.rho)
        
        #y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt =  alpha * (y[:,1] - y[:,0])
        dydt = y[:,0] * (rho - y[:,2]) - y[:,1]
        dzdt = y[:,0] * y[:,1] - beta * y[:,2]
        return torch.stack([dxdt, dydt, dzdt], dim = 1)
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        return  sigma + torch.zeros_like(y)



class repressilator(nn.Module):

    def __init__(self, alpha, beta_m, n, k, gamma_m, beta_p, gamma_p, sigma, propvola = True):
        super(repressilator, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta_m = nn.Parameter(torch.log(torch.tensor(beta_m)))
        self.beta_p = nn.Parameter(torch.log(torch.tensor(beta_p)))
        self.n = nn.Parameter(torch.log(torch.tensor(n)))
        self.k = nn.Parameter(torch.log(torch.tensor(k)))
        self.gamma_m = nn.Parameter(torch.log(torch.tensor(gamma_m)))
        self.gamma_p = nn.Parameter(torch.log(torch.tensor(gamma_p)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        alpha = self.preprocess(self.alpha)
        beta_m = self.preprocess(self.beta_m)
        beta_p = self.preprocess(self.beta_p)
        n = self.preprocess(self.n)
        k = self.preprocess(self.k)
        gamma_m = self.preprocess(self.gamma_m)
        gamma_p = self.preprocess(self.gamma_p)
        y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt = alpha + beta_m/(1.+ (y[:,2+3]/k) ** n) - gamma_m * y[:,0]
        dydt = alpha + beta_m/(1.+ (y[:,0+3]/k) ** n) - gamma_m * y[:,1]
        dzdt = alpha + beta_m/(1.+ (y[:,1+3]/k) ** n) - gamma_m * y[:,2]

        dxpdt = beta_p * y[:,0]-gamma_p * y[:,0+3]
        dypdt = beta_p * y[:,1]-gamma_p * y[:,1+3]
        dzpdt = beta_p * y[:,2]-gamma_p * y[:,2+3]
        return torch.stack([dxdt, dydt, dzdt, dxpdt, dypdt, dzpdt], dim = 1)
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        if self.propvola:
            return sigma * torch.relu(y)
        else:
            return sigma + torch.zeros_like(y)

class repressilator_mrna(nn.Module):
    def __init__(self, beta = 10, n = .3, k = 1, gamma = 1, sigma = .02, propvola = True):
        super(repressilator_mrna, self).__init__()
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.n = nn.Parameter(torch.log(torch.tensor(n)))
        self.k = nn.Parameter(torch.log(torch.tensor(k)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.propvola = propvola
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        
        beta = self.preprocess(self.beta)
        n = self.preprocess(self.n)
        k = self.preprocess(self.k)
        gamma = self.preprocess(self.gamma)
        y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt = beta/(1.+ (y[:,2]/k) ** n) - gamma * y[:,0]
        dydt = beta/(1.+ (y[:,0]/k) ** n) - gamma * y[:,1]
        dzdt = beta/(1.+ (y[:,1]/k) ** n) - gamma * y[:,2]
        return torch.stack([dxdt, dydt, dzdt], dim = 1)
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        if self.propvola:
            return sigma * torch.relu(y)
        else:
            return sigma + torch.zeros_like(y)

