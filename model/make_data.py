import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchsde
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from classic import LotkaVolterra, lorenz, repressilator_mrna
from missingobs import langevinlamboseen, repressilator_mrnaprotein
from GoM import GoM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def make_data(taskname):
    if "GoM" in taskname:
        gt = GoM(0.01)
        N_steps = 10 + 1
        dts = torch.tensor( 1.0*np.array([i for i in range(N_steps)])).to(device)/10. * 8.
        num_samples = 200
        X_0 = torch.rand(num_samples, 2)
        X_0[:,0] = X_0[:,0] * .1 - 0.05 
        X_0[:,1] = X_0[:,1] * .1 - 0.05 + (-0.75) 
        Xs = [None for _ in range(N_steps)]
        Xs[0] = X_0.to(device)
        
        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 2)
            X_0[:,0] = X_0[:,0] * .1 - 0.05 
            X_0[:,1] = X_0[:,1] * .1 - 0.05 + (-0.75) 
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]
            plt.scatter(Xs[i+1][:,0].cpu().detach().numpy(), 
                        Xs[i+1][:,1].cpu().detach().numpy())
    elif "LV" in taskname:
        gt = LotkaVolterra(1.0, 0.4, 0.4, 0.1, 0.02)
        N_steps = 10 + 1
        dts = torch.tensor( 1.0*np.array([i for i in range(N_steps)])).to(device)
        num_samples = 200
        X_0 = torch.rand(num_samples, 2)
        X_0[:,0] = X_0[:,0] * .1 + 5  # prey 
        X_0[:,1] = X_0[:,1] * .1 + 4 # predator
        Xs = [None for _ in range(N_steps)]
        Xs[0] = X_0.to(device)

        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 2)
            X_0[:,0] = X_0[:,0] * .1 + 5  # prey 
            X_0[:,1] = X_0[:,1] * .1 + 4 # predator
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]


        
    elif "Lorenz" in taskname:
        X0s = [2., 1., 1.] #[10., 10. , 10.] # initial point 
    
        lorenz_gt = lorenz(10., 28. ,8/3 , 1)
        lorenz_gt.to(device)
        N_steps = 10 + 1
        dts = torch.tensor( 1.0*np.array([i for i in range(N_steps)])/2).to(device) # until 3 works

        num_samples = 200
        X_0 = torch.rand(num_samples, 3)
        X_0[:,0] = X_0[:,0] * .1 + X0s[0] # confine the initial points   
        X_0[:,1] = X_0[:,1] * .1 + X0s[1] 
        X_0[:,2] = X_0[:,2] * .1 + X0s[2] 
        Xs = [None for _ in range(N_steps)]

        Xs[0] = X_0.to(device)

        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 3)
            X_0[:,0] = X_0[:,0] * .1 + X0s[0] # confine the initial points   
            X_0[:,1] = X_0[:,1] * .1 + X0s[1] 
            X_0[:,2] = X_0[:,2] * .1 + X0s[2]  
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(lorenz_gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]



    elif "Repressilatormrna" in taskname:
        repressilator_gt = repressilator_mrna(10.,3.,1.,1., 0.02)
        repressilator_gt.to(device)
        N_steps = 10 + 1
        dts = torch.tensor( 1.0*np.array([i for i in range(N_steps)])).to(device)

        num_samples = 200
        X_0 = torch.rand(num_samples, 3)
        X_0[:,0] = X_0[:,0] * .1 + 1   
        X_0[:,1] = X_0[:,1] * .1 + 1 
        X_0[:,2] = X_0[:,2] * .1 + 2 
        Xs = [None for _ in range(N_steps)]

        Xs[0] = X_0.to(device)

        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 3)
            X_0[:,0] = X_0[:,0] * .1 + 1   
            X_0[:,1] = X_0[:,1] * .1 + 1 
            X_0[:,2] = X_0[:,2] * .1 + 2 
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(repressilator_gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]

    elif "Repressilatormrnaprotein" in taskname:
        repressilator_gt = repressilator( alpha = 1e-5, 
                                 beta_m = 10.,
                                 n = 3., 
                                 k = 1., 
                                 gamma_m = 1., 
                                 beta_p = 1., 
                                 gamma_p = 1., 
                                 sigma = 0.02)
        repressilator_gt.to(device)
        N_steps = 10 + 1
        dts = torch.tensor( 1.0*np.array([i for i in range(N_steps)])).to(device)

        num_samples = 200
        X_0 = torch.rand(num_samples, 6)
        X_0[:,0] = X_0[:,0] * .1 + 1   
        X_0[:,1] = X_0[:,1] * .1 + 1 
        X_0[:,2] = X_0[:,2] * .1 + 2. 
        X_0[:,0 + 3] = X_0[:,0 + 3] * .1 + 0   
        X_0[:,1 + 3] = X_0[:,1 + 3] * .1 + 0 
        X_0[:,2 + 3] = X_0[:,2 + 3] * .1 + 0 
        Xs = [None for _ in range(N_steps)]

        Xs[0] = X_0[:,:3].to(device) # see only mRNA

        for i in tqdm(range(N_steps-1)):
            X_0 = torch.rand(num_samples, 6)
            X_0[:,0] = X_0[:,0] * .1 + 1   
            X_0[:,1] = X_0[:,1] * .1 + 1 
            X_0[:,2] = X_0[:,2] * .1 + 2 
            X_0[:,0 + 3] = X_0[:,0 + 3] * .1 + 0   
            X_0[:,1 + 3] = X_0[:,1 + 3] * .1 + 0 
            X_0[:,2 + 3] = X_0[:,2 + 3] * .1 + 0
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(repressilator_gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1][:,:3]


    
    elif "vortex" in taskname:
        langevinlamboseen_gt = langevinlamboseen(x0 = 0., y0 = 0., 
                                                 logscale = -2., 
                                                 circulation = 1., 
                                                 logsigma = np.log(0.02), 
                                                 logdrag = np.log(0.5)
                                                 )
        langevinlamboseen_gt.to(device)
        N_steps = 10 + 1
        maxtime = 8.
        dts = torch.tensor( 1.0*np.array([i for i in range(N_steps)])).to(device)/N_steps * maxtime
        num_samples = 200
        X_0 = torch.rand(num_samples, 4)
        X_0[:,0] = X_0[:,0] * .1 - 0.05 - 1
        X_0[:,1] = X_0[:,1] * .1 - 0.05  - 1
        X_0[:,2] = X_0[:,2] * .01 - 0.005
        X_0[:,3] = X_0[:,3] * .01 - 0.005
        Xs = [None for _ in range(N_steps)]
        Xs[0] = X_0[:,:2].to(device)

        for i in tqdm(range(N_steps-1)):
            X_0 = torch.rand(num_samples, 4)
            X_0[:,0] = X_0[:,0] * .1 - 0.05 - 1
            X_0[:,1] = X_0[:,1] * .1 - 0.05  - 1
            X_0[:,2] = X_0[:,2] * .01 - 0.005
            X_0[:,3] = X_0[:,3] * .01 - 0.005
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(langevinlamboseen_gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')
            Xs[i+1] = ys[-1][:,:2]
            plt.scatter(Xs[i+1][:,0].cpu().detach().numpy(), 
                        Xs[i+1][:,1].cpu().detach().numpy())


    filename = f"../asset/{taskname}_data.npz"
    np.savez(filename, 
             N_steps = N_steps,
             Xs = torch.stack(Xs).cpu().detach().numpy(),
             dts = dts.cpu().detach().numpy())





def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <your_argument>")
        sys.exit(1)
    
    task = sys.argv[1]
    make_data(task)

if __name__ == "__main__":
    main()
