import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torchdiffeq import odeint
from typing import Tuple
ACTIVATION_FN = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}

class MLPVectorField(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = 'leaky_relu',
        residual: bool = False,
        metric=None,  # Optional Riemann metric for computing Riemannian energy
        unbalanced: bool = False,
        alpha_growth: float = 1.0,
    ):
        super().__init__()
        self.unbalanced = unbalanced
        self.alpha_growth = alpha_growth

        self.vector_field_net = HyperNetwork(
            input_dim=dim + 1,
            output_dim=dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            residual=residual,
        )
        self.metric = metric  # Store metric for Riemannian energy computation

        if self.unbalanced:
            # Growth network: input (z, t) -> scalar growth rate g
            self.growth_net = HyperNetwork(
                input_dim=dim + 1,
                output_dim=1,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                activation=activation,
                residual=residual,
            )

    def forward(self, t: torch.Tensor, states):
        """
        t: scalar tensor
        z: (batch, dim)

        Balanced mode:
            states: (z, e)
            Returns: (dz_dt, de_dt)

        Unbalanced mode:
            states: (z, lnw, e_velocity, e_growth)
            Returns: (dz_dt, dlnw_dt, de_velocity_dt, de_growth_dt)
            where dlnw_dt = g (growth rate)
            de_velocity_dt = 0.5 * ||v||^2 * exp(lnw)
            de_growth_dt = alpha_growth * g^2 * exp(lnw)
        """
        z = states[0]
        batchsize = z.shape[0]

        # broadcast t
        t = t.expand(batchsize, 1)
        inp = torch.cat([z, t], dim=1)

        dz_dt = self.vector_field_net(inp)

        if self.unbalanced:
            lnw = states[1]
            g = self.growth_net(inp)  # (batch, 1)
            dlnw_dt = g * self.alpha_growth
            w = torch.exp(lnw)

            # Velocity energy
            if self.metric is not None:
                riemannian_speed = self.metric.speed(z, dz_dt)  # (batch,)
                v_energy = 0.5 * riemannian_speed.unsqueeze(-1)  # (batch, 1)
            else:
                v_energy = 0.5 * (dz_dt ** 2).sum(dim=1, keepdim=True)

            de_velocity_dt = v_energy * w
            de_growth_dt = 0.5 * self.alpha_growth * (g ** 2) * w

            return dz_dt, dlnw_dt, de_velocity_dt, de_growth_dt
        else:
            # Balanced mode (original behavior)
            if self.metric is not None:
                riemannian_speed = self.metric.speed(z, dz_dt)  # (batch,)
                de_dt = 0.5 * riemannian_speed.unsqueeze(-1)    # (batch, 1)
            else:
                de_dt = 0.5 * (dz_dt ** 2).sum(dim=1, keepdim=True)

            return dz_dt, de_dt

class MLPScoreFunction(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = 'leaky_relu',
        residual: bool = False,
    ):
        super().__init__()

        self.net = HyperNetwork(
            input_dim=dim + 1,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            residual=residual,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        """
        t: scalar tensor
        z: (batch, dim)
        """
        batchsize = z.shape[0]

        # broadcast t
        t = t.expand(batchsize, 1)
        inp = torch.cat([z, t], dim=1)

        dz_dt = self.net(inp)
        return dz_dt

class HyperNetwork(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 400,
            n_layers: int = 2,
            activation: str = 'leaky_relu',
            residual: bool = False
    ):
        super().__init__()

        if activation not in ACTIVATION_FN:
            raise ValueError(f"Activation '{activation}' not recognized.")

        self.n_layers = n_layers
        self.residual = residual
        act_fn = ACTIVATION_FN[activation]

        if self.n_layers == 0:
            self.input_layer = nn.Linear(input_dim, output_dim)
            self.hidden_layers = nn.ModuleList([])
            self.output_layer = nn.Identity()
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn()
            )

            self.hidden_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        act_fn()
                    )
                    for _ in range(n_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers == 0:
            return self.input_layer(x)

        x = self.input_layer(x)

        for layer in self.hidden_layers:
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)

        x = self.output_layer(x)
        return x
class CNF(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.n_hiddens = n_hiddens
        self.activation = activation
        self.width = hidden_dim * 2

        self.balanced_net = BalancedNetwork(in_out_dim, hidden_dim, width=self.width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
                
            W, B, U = self.balanced_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)
            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)           
            #dz_dt = self.balanced_net(t, z)
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
            
            return (dz_dt, dlogp_z_dt)


def trace_df_dz(f, z):

    sum_diag = 0.0
    for i in range(z.shape[1]):
     
        sum_diag += torch.autograd.grad(f[:, i].sum(),
                                        z, 
                                        create_graph=True)[0].contiguous()[:, i].contiguous()
        
    return sum_diag.contiguous()

class BalancedNetwork(nn.Module):

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, t):
        
        params = t.reshape(1, 1)
        params = self.leaky_relu(self.fc1(params))
        params = self.leaky_relu(self.fc2(params))
        params = self.fc3(params)

        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        
        return [W, B, U]
    
def compute_vector_field(
    model,
    z: torch.Tensor,     # (N, d)
    t: float,
    MLPModel: bool = True,
    unbalanced: bool = False,
    device: str = 'cpu',
):
    z = z.to(device)
    e0 = torch.zeros(z.shape[0], 1, device=device)
    model = model.to(device)
    t_tensor = torch.tensor(t, dtype=z.dtype, device=device)

    if unbalanced:
        lnw = torch.zeros(z.shape[0], 1, device=device)
        dz_dt, dlnw_dt, de_velocity_dt, de_growth_dt = model(t_tensor, (z, lnw, e0, e0))
        return dz_dt, dlnw_dt
    else:
        logp = torch.zeros(z.shape[0], 1, device=device)
        if MLPModel:
            dz_dt, _ = model(t_tensor, (z, e0))
        else:
            dz_dt, _ = model(t_tensor, (z, logp))
        return dz_dt, None


    
class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()     

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]   
        t = t.clone().detach().to(x.device).repeat(batchsize).reshape(-1, 1)                    
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x


class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
        
    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]   
        t = t.clone().detach().to(x.device).repeat(batchsize).reshape(-1, 1)            
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
    
    

class RunningAverageMeter(object):

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
def ggrowth(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)                       
    gg = func.forward(t, y)[2]
    return (y_0,y_00,gg)

