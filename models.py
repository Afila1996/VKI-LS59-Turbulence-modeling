from datasets import load_dataset
import pickle
from plaid.containers.sample import Sample
import numpy as np
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score


class LpLoss(nn.Module):

    def __init__(self, p=2, reduction='mean'):
        super(LpLoss, self).__init__()
        self.p         = p
        self.reduction = reduction

    def forward(self, pred, target):

        B = pred.shape[0]


        pred_flat   = pred.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        diff_norm   = torch.norm(pred_flat - target_flat, p=self.p, dim=1)
        target_norm = torch.norm(target_flat,             p=self.p, dim=1) + 1e-8

        loss = diff_norm / target_norm

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GaussianNormalizer(nn.Module):

    def __init__(self, x, eps=1e-8):
        super(GaussianNormalizer, self).__init__()

        self.register_buffer('mean', x.mean(dim=(0, 2, 3), keepdim=True))
        self.register_buffer('std',  x.std( dim=(0, 2, 3), keepdim=True))
        self.eps = eps

    def encode(self, x):

        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):

        return x * (self.std + self.eps) + self.mean



class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1
        self.modes2       = modes2

        # He initialisation
        fan_in = in_channels * modes1 * modes2
        std    = (2.0 / fan_in) ** 0.5
        scale  = std / (2.0 ** 0.5)

        self.weights1 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(in_channels, out_channels, modes1, modes2, 2) * scale
            )
        )
        self.weights2 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(in_channels, out_channels, modes1, modes2, 2) * scale
            )
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        ## FFT
        x_ft = torch.fft.rfft2(x)


        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)


        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],
                             self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                             self.weights2)

        ## INVERSE FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x



class iDAFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, nlayer,
                 inp_size=5, out_size=5):
        super(iDAFNO2d, self).__init__()


        self.modes1   = modes1
        self.modes2   = modes2
        self.width    = width
        self.nlayer   = nlayer
        self.inp_size = inp_size
        self.out_size = out_size


        self.fc0 = nn.Linear(self.inp_size, self.width)


        self.convlayer = nn.ModuleList([
            SpectralConv2d(self.width, self.width,
                           self.modes1, self.modes2)
            for _ in range(1)
        ])


        self.w = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(1)
        ])

        self.fc1 = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, self.out_size)
        )

    def forward(self, x, chi):

        batchsize = x.shape[0]
        size_x    = x.shape[2]
        size_y    = x.shape[3]


        chi_expand = chi.expand(batchsize, self.width, size_x, size_y) #(b,32,64,64)


        x = x.permute(0, 2, 3, 1) #(b,64,64,5)


        x = self.fc0(x) #(b,64,64,32)


        x = x.permute(0, 3, 1, 2) #(b,32,64,64)

        coef     = 1.0 / self.nlayer
        conv_chi = self.convlayer[0](chi_expand)



        for _ in range(self.nlayer - 1):


            conv_chix = self.convlayer[0](chi_expand * x)


            xconv_chi = x * conv_chi


            wx = self.w[0](x)


            x = F.gelu(chi_expand * (conv_chix - xconv_chi + wx)) * coef + x


        conv_chix = self.convlayer[0](chi_expand * x)
        xconv_chi = x * conv_chi
        wx        = self.w[0](x)


        x = chi_expand * (conv_chix - xconv_chi + wx) * coef + x


        x = x.permute(0, 2, 3, 1)


        x = self.fc1(x)


        x = x.permute(0, 3, 1, 2)


        return x


##Physics functions

def finite_difference_derivative(field, direction):

    if direction == 'x':
        dim = 2
        N   = field.shape[2]
    else:
        dim = 1
        N   = field.shape[1]

    dx = 1.0 / (N - 1)
    d  = torch.zeros_like(field)

    if direction == 'x':
        d[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * dx)
        d[:, :, 0]    = (field[:, :, 1]  - field[:, :, 0])   / dx
        d[:, :, -1]   = (field[:, :, -1] - field[:, :, -2])  / dx
    else:
        d[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dx)
        d[:, 0,    :] = (field[:, 1,  :] - field[:, 0,   :]) / dx
        d[:, -1,   :] = (field[:, -1, :] - field[:, -2,  :]) / dx

    return d

def compute_euler_residuals(fields_phys, clamp_val=10.0, gamma=1.4):
    gamma=1.4

    ro  = fields_phys[:, 0]
    rou = fields_phys[:, 1]
    rov = fields_phys[:, 2]
    roe = fields_phys[:, 3]


    ro_safe = ro.abs()  + 1e-8
    u       = rou / ro_safe
    v       = rov / ro_safe
    p       = (gamma - 1.0) * (roe - 0.5*(rou**2 + rov**2) / ro_safe)
    p_safe  = p.abs() + 1e-8


    R_mass = (finite_difference_derivative(rou, 'x') +
              finite_difference_derivative(rov, 'y'))


    R_momx = (finite_difference_derivative(rou**2/ro_safe + p_safe, 'x') +
              finite_difference_derivative(rou*rov/ro_safe,          'y'))


    R_momy = (finite_difference_derivative(rou*rov/ro_safe,          'x') +
              finite_difference_derivative(rov**2/ro_safe + p_safe,  'y'))


    R_ene  = (finite_difference_derivative((roe + p_safe) * u, 'x') +
              finite_difference_derivative((roe + p_safe) * v, 'y'))


    R_mass = torch.clamp(R_mass, -clamp_val, clamp_val)
    R_momx = torch.clamp(R_momx, -clamp_val, clamp_val)
    R_momy = torch.clamp(R_momy, -clamp_val, clamp_val)
    R_ene  = torch.clamp(R_ene,  -clamp_val, clamp_val)

    return R_mass, R_momx, R_momy, R_ene


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_RS_targets(Y_phys_np, batch_size=32, device=device):

    Y_tensor = torch.tensor(Y_phys_np, dtype=torch.float32).permute(0,3,1,2)
    N        = Y_tensor.shape[0]
    lists    = [[], [], [], []]

    for i in range(0, N, batch_size):
        batch = Y_tensor[i:i+batch_size].to(device)
        with torch.no_grad():
            Rs = compute_euler_residuals(batch)
        for k, R in enumerate(Rs):
            lists[k].append(R.cpu())

    return [torch.cat(l, dim=0) for l in lists]



RS_tr_mass,  RS_tr_momx,  RS_tr_momy,  RS_tr_ene  = precompute_RS_targets(Y_tr_raw)
RS_val_mass, RS_val_momx, RS_val_momy, RS_val_ene = precompute_RS_targets(Y_val_raw)


SCALE_MASS = float(RS_tr_mass.std() ** 2) + 1e-8
SCALE_MOMX = float(RS_tr_momx.std() ** 2) + 1e-8
SCALE_MOMY = float(RS_tr_momy.std() ** 2) + 1e-8
SCALE_ENE  = float(RS_tr_ene.std()  ** 2) + 1e-8



def rans_physics_loss(pred_phys,
                      rs_mass_gt,
                      rs_momx_gt,
                      rs_momy_gt,
                      rs_ene_gt,
                      w_mass    = 0.0,
                      w_momx    = 0.5,
                      w_momy    = 0.5,
                      w_ene     = 2.0,
                      clamp_val = 10.0):

    R_mass_p, R_momx_p, R_momy_p, R_ene_p = compute_euler_residuals(
        pred_phys, clamp_val=clamp_val
    )


    def prep(t):
        return torch.clamp(t.to(pred_phys.device), -clamp_val, clamp_val)


    l_mass = ((R_mass_p - prep(rs_mass_gt))**2).mean() / SCALE_MASS
    l_momx = ((R_momx_p - prep(rs_momx_gt))**2).mean() / SCALE_MOMX
    l_momy = ((R_momy_p - prep(rs_momy_gt))**2).mean() / SCALE_MOMY
    l_ene  = ((R_ene_p  - prep(rs_ene_gt ))**2).mean() / SCALE_ENE

    loss_phys = (w_mass * l_mass +
                 w_momx * l_momx +
                 w_momy * l_momy +
                 w_ene  * l_ene)

    return loss_phys, l_mass, l_momx, l_momy, l_ene



