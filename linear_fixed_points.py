import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from torch.func import jacrev
from pcgrad import PCGrad
import pandas as pd
import seaborn as sns
import sympy as sp
torch.set_default_device('cuda')

def get_sol(A, interval, c_i):
    def xdot(t, x):
        return A @ x
    sol = solve_ivp(xdot, interval, c_i, t_eval=np.linspace(interval[0], interval[1], 250))
    return np.concatenate([sol.t.reshape((1, sol.t.shape[0])), sol.y[0:1], sol.y[1:2]])
    
def true_trajectory_A(A: list[list], interval: list[int], c_i: list[int], ax=None, color='black') -> None:
    sol = get_sol(A, interval, c_i)
    if ax:
        ax.plot(sol[1, :], sol[2, :], label="Trajectoire ("+r"$x(0)$"+f" = {c_i})", linestyle='dashed', color=color)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    else:
        plt.plot(sol[1, :], sol[2, :], label="Trajectoire ("+r"$x(0)$"+f" = {c_i})", linestyle='dashed', color=color)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")

def true_phase_portrait_A(A, ax=None):
    x1, x2 = np.meshgrid(np.arange(-1.5, 1.6, 0.2), np.arange(-1.5, 1.6, 0.2))
    x1_values = A[0,0]*x1 + A[0,1]*x2
    x2_values = A[1,0]*x1 + A[1,1]*x2

    norms = np.sqrt(x1_values**2 + x2_values**2)
    x1_normalized = x1_values / norms
    x2_normalized = x2_values / norms

    vector_scale = 25
    if not ax:
        plt.quiver(x1, x2, x1_normalized, x2_normalized, scale=vector_scale)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.grid()
    else:
        ax.quiver(x1, x2, x1_normalized, x2_normalized, scale=vector_scale)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_title("Reference")

n_points = 50 #00
n_epochs = 10 #000

T_MAX = 3.
#A = np.array([[1, 1], [-1, 1]])
#c_is = [[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5,0.5]]

#plt.figure(figsize=(10, 10))
#for c_i in c_is:
#    true_trajectory_A(A, (0, T_MAX), c_i)

#true_phase_portrait_A(A)

class XSinLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x) * x
        
class TransitionPINN_1(nn.Module):
    def __init__(self, layer_size=256):
        super().__init__()
        self.l1 = nn.Linear(4, layer_size)
        self.l2 = nn.Linear(layer_size, layer_size)
        self.l3 = nn.Linear(layer_size, layer_size)
        self.l4 = nn.Linear(layer_size, layer_size)
        self.l5 = nn.Linear(layer_size, layer_size)
        self.l6 = nn.Linear(layer_size, 2)
        self.activation = XSinLayer()

    def forward(self, t, t0, x1_0, x2_0):
        inputs = torch.concatenate([t, t0, x1_0, x2_0], axis=-1)
        x = self.activation(self.l1(inputs))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.l5(x))
        return (t-t0) * self.l6(x) + torch.concatenate([x1_0, x2_0], axis=-1)

def loss_physics_A(A, model, t, t0, x1_0, x2_0):
    x1x2 = model(t, t0, x1_0, x2_0)
    x1, x2 = x1x2[:, 0:1], x1x2[:, 1:2]
    dx1_dt = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    dx2_dt = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    loss_x1 = ((dx1_dt - (A[0,0]*x1 + A[0,1]*x2))**2).mean()
    loss_x2 = ((dx2_dt - (A[1,0]*x1 + A[1,1]*x2))**2).mean()
    return loss_x1 + loss_x2

def train_model_1(A, seed: int):
    torch.random.manual_seed(seed)
    model_A = TransitionPINN_1()
    optimizer = optim.Adam(model_A.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, min_lr=1e-5)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        t0 = torch.rand((n_points, 1), requires_grad=True) * T_MAX
        t = t0 + torch.rand((n_points, 1), requires_grad=True) * (T_MAX-t0)
        x1_0 = (torch.rand((n_points, 1), requires_grad=True)-0.5) * 8.
        x2_0 = (torch.rand((n_points, 1), requires_grad=True)-0.5) * 8.
        loss = loss_physics_A(A, model_A, t, t0, x1_0, x2_0)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
    print(loss.item())
    return model_A

def model_trajectory(model, interval: list[int], c_i: list[int], ax=None, color='black') -> None:
    t_eval = torch.linspace(interval[0], interval[1], 250).unsqueeze(-1)
    x1x2 = model(t_eval, torch.zeros_like(t_eval), torch.ones_like(t_eval)*c_i[0], torch.ones_like(t_eval)*c_i[1]).detach().cpu().squeeze().numpy()
    x1, x2 = x1x2[:, 0], x1x2[:, 1]
    if ax:
        ax.plot(x1, x2, label="Trajectoire ("+r"$x(0)$"+f" = {c_i})", color=color)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    else:
        plt.plot(x1, x2, label="Trajectoire ("+r"$x(0)$"+f" = {c_i})", color=color)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")

def model_phase_portrait(model, ax, t=1., opacity=1.):
    x1_0, x2_0 = torch.meshgrid(torch.arange(-1.5, 1.6, 0.2), torch.arange(-1.5, 1.6, 0.2))
    x1_0 = x1_0.flatten().unsqueeze(-1)
    x2_0 = x2_0.flatten().unsqueeze(-1)
    t = torch.ones_like(x1_0, requires_grad=True)*t
    x1x2 = model(t, t.detach(), x1_0, x2_0)
    x1, x2 = x1x2[:, 0:1], x1x2[:, 1:2]
    x1_values = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(t), create_graph=True)[0].cpu().detach().squeeze().numpy()
    x2_values = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(t), create_graph=True)[0].cpu().detach().squeeze().numpy()

    norms = np.sqrt(x1_values**2 + x2_values**2)
    x1_normalized = x1_values / norms
    x2_normalized = x2_values / norms

    vector_scale = 25
    ax.quiver(x1_0.cpu().detach().squeeze(), x2_0.cpu().detach().squeeze(), x1_normalized, x2_normalized, scale=vector_scale, alpha=opacity)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_title(f"t: {t[0].item()}")

def display_results(model):
    c_i1, c_i2 = np.meshgrid(np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5))
    c_is = (np.array(list(zip(c_i1.flatten(), c_i2.flatten()))))
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    colors = plt.cm.magma(np.linspace(0, 1, len(c_is)))

    for c_i, color in zip(c_is, colors):
        model_trajectory(model, (0, T_MAX), c_i, ax=ax[0], color=color)
        true_trajectory_A(A, (0, T_MAX), c_i, ax=ax[0], color=color)
    ax[0].set_xlim(-1.5, 1.5)
    ax[0].set_ylim(-1.5, 1.5)
    ax[0].grid()
    for t in np.arange(0, T_MAX, .5):
        model_phase_portrait(model, ax[1], t, opacity=t/T_MAX)
    ax[1].set_title("Phase portrait of the model through time")
    ax[1].legend([f"t={t}" for t in np.arange(0, T_MAX, .5)])
    true_phase_portrait_A(A, ax[2])
    model_phase_portrait(model, ax[2], 0, opacity=0.5)
    ax[2].legend(["Reference", "Model at t=0"])
    ax[2].set_title("Reference and model")

class TransitionPINN_2(nn.Module):
    def __init__(self, layer_size=256):
        super().__init__()
        self.l1 = nn.Linear(4, layer_size)
        self.l2 = nn.Linear(layer_size, layer_size)
        self.l3 = nn.Linear(layer_size, layer_size)
        self.l4 = nn.Linear(layer_size, layer_size)
        self.l5 = nn.Linear(layer_size, layer_size)
        self.l6 = nn.Linear(layer_size, 2)
        self.activation = XSinLayer()

    def forward(self, t, t0, x1_0, x2_0):
        tau = t - t0
        inputs = torch.concatenate([tau, tau**2, x1_0, x2_0], axis=-1)
        x = self.activation(self.l1(inputs))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.l5(x))
        return tau * self.l6(x) + torch.concatenate([x1_0, x2_0], axis=-1)


def consistency_loss(model, t0, x1_0, x2_0):
    x1x2 = model(t0, t0, x1_0, x2_0)
    return ((x1x2-torch.concatenate([x1_0, x2_0], axis=-1))**2).mean()

def composition_loss(model, t, t0, t1, x1_0, x2_0):
    x1 = model(t, t0, x1_0, x2_0)
    x1x2_1 = model(t1, t0, x1_0, x2_0)
    x1_1, x2_1 = x1x2_1[:, 0:1].detach(), x1x2_1[:, 1:2].detach()
    x2 = model(t, t1, x1_1, x2_1)
    return ((x2-x1)**2).mean()

def stationarity_loss(model, t, t0, x1_0, x2_0, dt):
    x1x2 = model(t, t0, x1_0, x2_0)
    x1x2_dt = model(t+dt, t0+dt, x1_0, x2_0)
    return ((x1x2-x1x2_dt)**2).mean()

def derivative_stationarity_loss(model, t, t0, x1_0, x2_0, dt):
    x1x2 = model(tau, x1_0, x2_0)
    x1, x2 = x1x2[:, 0:1], x1x2[:, 1:2]
    x1x2_dt = model(t+dt, t0+dt, x1_0, x2_0)
    x1_dt, x2_dt = x1x2_dt[:, 0:1], x1x2_dt[:, 1:2]
    dx1 = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    dx1dt = torch.autograd.grad(x1_dt, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    dx2 = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    dx2dt = torch.autograd.grad(x2_dt, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    return ((dx1-dx1dt)**2).mean() + ((dx2-dx2dt)**2).mean()

def loss_physics_A2(A, model, t, t0, x1_0, x2_0):
    x1x2 = model(t, t0, x1_0, x2_0)
    x1, x2 = x1x2[:, 0:1], x1x2[:, 1:2]
    dx1_dt = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    dx2_dt = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    loss_x1 = ((dx1_dt - (A[0,0]*x1 + A[0,1]*x2))**2).mean()
    loss_x2 = ((dx2_dt - (A[1,0]*x1 + A[1,1]*x2))**2).mean()
    return loss_x1 + loss_x2

def train_model_2(A, seed: int):
    torch.random.manual_seed(seed)
    model_A2 = TransitionPINN_2()
    optimizer = optim.Adam(model_A2.parameters(), lr=1e-3)
    #optimizer = PCGrad(op)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, min_lr=1e-5)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        t0 = torch.rand((n_points, 1), requires_grad=True) * T_MAX
        t = t0 + torch.rand((n_points, 1), requires_grad=True) * (T_MAX-t0)
        x1_0 = (torch.rand((n_points, 1), requires_grad=True)-0.5) * 8.
        x2_0 = (torch.rand((n_points, 1), requires_grad=True)-0.5) * 8.
        loss1 = loss_physics_A2(A, model_A2, t, t0, x1_0, x2_0)
        t1 = t0 + torch.rand((n_points, 1), requires_grad=True) * (t-t0)
        #dt = (torch.rand((n_points, 1)))
        #loss2 = consistency_loss(model_A2, t0, x1_0, x2_0)
        loss3 = composition_loss(model_A2, t, t0, t1, x1_0, x2_0)
        #loss4 = stationarity_loss(model_A2, t, t0, x1_0, x2_0, dt)
        #loss5 = derivative_stationarity_loss(model_A2, t, t0, x1_0, x2_0, dt)
        #losses = [loss1 + loss3] #,loss4,loss5]
        #losses.backward()
        #optimizer.pc_backward(losses)
        loss = loss1 + loss3
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
    print(loss1.item())
    return model_A2

def get_references(A):
    references = []
    np.random.seed(0)
    T_MAX_REFERENCE = 10.
    N_POINTS_REFERENCE = 1000
    c_is_reference = (np.random.random(size=(N_POINTS_REFERENCE, 2))-0.5) * 50
    for c_i in c_is_reference:
        t0 = np.random.random()*T_MAX_REFERENCE
        references.append(get_sol(A, (t0, T_MAX_REFERENCE+t0), c_i))
    return torch.tensor(np.array(references), dtype=torch.float32)

def evaluate_model_on_trajectories(model, references):
    error = 0
    with torch.no_grad():
        for i in range(references.shape[0]):
            t = references[i, 0, :].unsqueeze(1)
            t0 = references[i, 0, 0:1] * torch.ones_like(t)
            x1_0 = references[i, 1, 0:1] * torch.ones_like(t)
            x2_0 = references[i, 2, 0:1] * torch.ones_like(t)
            y_hat = model(t, t0, x1_0, x2_0)
            y = references[i, 1:].T
            error += ((y_hat-y).abs()).mean()
    return error.item()/references.shape[0]

def g_at_tau0(model, x1_0, x2_0):
    ones = torch.ones_like(x1_0)
    return model(ones, 0*ones, x1_0, x2_0).cpu()

def residual_np(xy, model, t0_scalar):
    x1_0 = torch.tensor([[xy[0]]], dtype=torch.float32)
    x2_0 = torch.tensor([[xy[1]]], dtype=torch.float32)
    with torch.no_grad():
        F = g_at_tau0(model, x1_0, x2_0)
    return F.squeeze(0).numpy()

def get_distance_with_true_fixed_point(model):
    root = fsolve(residual_np, x0=[0, 0], args=(model, 0.0))
    return (root[0]**2 + root[1]**2)**0.5

n_trials = 5

As = []
models_1 = []
models_2 = []

for seed in tqdm(range(n_trials)):
    np.random.seed(seed)
    As.append(np.random.random((2, 2))-0.5)
    models_1.append(train_model_1(As[seed], seed))
    models_2.append(train_model_2(As[seed], seed))


def classify_equilibrium(A):
    """
    Returns:
        0: stable
        1: unstable
        2: saddle
    """
    tr = np.trace(A)
    det = np.linalg.det(A)

    if det < 0:
        return 2  # saddle

    elif tr < 0:
        return 0  # stable

    else:
        return 1  # unstable

errors = {'Model 1': [], 'Model 2': []}
fixed_point_distances = {'Model 1': [], 'Model 2': []}
stabilities = []

for seed in range(n_trials):
    references = get_references(As[seed])

    errors['Model 1'].append(
        evaluate_model_on_trajectories(models_1[seed], references)
    )
    errors['Model 2'].append(
        evaluate_model_on_trajectories(models_2[seed], references)
    )

    fixed_point_distances['Model 1'].append(
        get_distance_with_true_fixed_point(models_1[seed])
    )
    fixed_point_distances['Model 2'].append(
        get_distance_with_true_fixed_point(models_2[seed])
    )

    stabilities.append(classify_equilibrium(As[seed]))

df_plot = pd.DataFrame({
    "Stability": stabilities,
    "Model 1": fixed_point_distances["Model 1"],
    "Model 2": fixed_point_distances["Model 2"],
})

df_plot_long = df_plot.melt(
    id_vars="Stability",
    value_vars=["Model 1", "Model 2"],
    var_name="Model",
    value_name="Distance"
)

df_plot_long.to_csv("pinn_linear_fixed_points.csv")

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_plot_long,
    x="Stability",
    y="Distance",
    hue="Model",
    order=[0, 1, 2]
)
plt.xlabel("Stability")
plt.ylabel("Distance to true fixed point")
plt.xticks([0, 1, 2], ["Stable", "Unstable", "Saddle"])
plt.grid()
plt.tight_layout()
plt.savefig("pinn_linear_fixed_points.jpg", dpi=300)
plt.close()
