import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchsde
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_GoMvortex():
    df = pd.read_csv("../asset/hycom_GoM_part1.csv")
    df1 = pd.read_csv("../asset/hycom_GoM_part2.csv")
    df = pd.concate([df, df1])
    min_lat = 24
    min_lon = -92.5
    max_lat = 27
    max_lon = -89
    center = [-90.5, 25.75]

    df_vortex_sub = df[(df['x'] >= min_lon) & (df['x'] <= max_lon) & (df['y'] >= min_lat) & (df['y'] <= max_lat)]


    x_grid = df_vortex_sub['x'].values - center[0]
    y_grid = df_vortex_sub['y'].values - center[1] # center a bit
    u_grid = df_vortex_sub['u'].values
    v_grid = df_vortex_sub['v'].values

    grid_points = np.column_stack((x_grid, y_grid))
    tree = cKDTree(grid_points)
    def get_velocity(y):
        y_np = y.detach().cpu().numpy()
        #breakpoint()
        _, idx = tree.query(y_np.tolist())  # Find the nearest point on the grid
        return  torch.stack([torch.tensor(u_grid[idx]), torch.tensor(v_grid[idx])], axis=1).to(y.device)

    return get_velocity


class GoM(nn.Module):
    def __init__(self, sigma = 0.01, vectorfield = get_GoMvortex()):
        super(GoM, self).__init__()
        self.vectorfield = vectorfield
        
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        return self.vectorfield(y)
        
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        return  sigma + 0 * torch.relu(y)
