import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.channel_wise = channel_wise
        self.is_act = is_act
        self.register_buffer('inited', torch.zeros(1))
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={}, bit = {})".format(self.inited, self.channel_wise, self.n_bits)
        return s

    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
            self.inited.fill_(1)

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if self.is_act:
                if len(x.shape) == 3:
                    n_channels = x_clone.shape[-1]
                elif len(x.shape) == 4: 
                    n_channels = x_clone.shape[1] 
                elif len(x.shape) == 2: 
                    n_channels = x_clone.shape[1]
                else:
                    raise NotImplementedError

                if len(x.shape) == 4: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=0)[0]
                elif len(x.shape) == 3: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,...], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(1, -1, 1, 1)
                    zero_point = zero_point.view(1, -1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(1, -1)
                    zero_point = zero_point.view(1, -1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
                
            else: 
                n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(-1, 1)
                    zero_point = zero_point.view(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            if self.is_act:
                search_range = [0.999, 0.9999, 0.99999]
            else:
                search_range = [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999]
            for pct in search_range:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

class SoftmaxLogQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act: bool = False):
        super(SoftmaxLogQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.taus = None
        self.register_buffer('inited', torch.zeros(1))
        self.channel_wise = channel_wise
        self.is_act = is_act

    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            delta , taus = self.init_quantization_scale(x)
            self.delta = delta
            self.taus = taus
            self.inited.fill_(1)
        x = x + self.delta
        x_int = round_ste(-1 * (x / self.delta1).log2() * self.taus) #self.delta是偏置，self.detal1是量化尺度
        mask = x_int >= 2 ** (self.n_bits)
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        n = torch.floor(-1 * (x_quant / self.taus))
        m = (-x_quant % self.taus) / self.taus
        x_float_q = self.delta1 * (2 ** n) * 2 ** m
        x_float_q = x_float_q - self.delta
        x_float_q[mask] = 0
        return x_float_q

    def init_quantization_scale(self, x: torch.Tensor):
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for i in range(self.n_bits):
            taus = torch.tensor(2 ** i)
            for bias in [0,0.0001,0.0005,0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
                 x_q = self.quantize(x_clone,delta,taus,bias)
                 score = lp_loss(x_clone, x_q, p=2, reduction='all')
                 if score < best_score:
                     best_score = score
                     cur_bais = bias
                     new_taus= taus
                     self.delta1 = delta
        print(self.delta1)
        print(new_taus)
        print(cur_bais)
        return cur_bais, new_taus


    def quantize(self, x, delta,taus,bias):
        x1 = x + bias
        x_int = round_ste(-1 * (x1 / delta).log2() * taus)
        mask = x_int >= 2 ** (self.n_bits)
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        n = torch.floor(-1 * (x_quant / taus))
        m = (-x_quant % taus) / taus
        x_float_q = delta * (2 ** n) * 2 ** m
        x_float_q = x_float_q -bias
        x_float_q[mask] = 0
        return x_float_q
class GELULogQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act: bool = False):
        super(GELULogQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.taus = None
        self.register_buffer('inited', torch.zeros(1))
        self.channel_wise = channel_wise
        self.is_act = is_act

    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            delta , taus = self.init_quantization_scale(x)
            self.delta = delta
            self.taus = taus
            self.inited.fill_(1)
        x = x +0.17 + self.delta
        x_int = round_ste(-1 * (x / self.delta1).log2() * self.taus)
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        n = torch.floor(-1 * (x_quant / self.taus))
        m = (-x_quant % self.taus) / self.taus
        x_float_q = self.delta1 * (2 ** n) * 2 ** m
        x_float_q = x_float_q - self.delta -0.17
        return x_float_q

    def init_quantization_scale(self, x: torch.Tensor):
        x_clone = x.clone().detach()
        x_clone = x_clone + 0.17
        delta = x_clone.max()
        best_score = 1e+10
        for i in range( self.n_bits):
            taus = torch.tensor(2 ** i)
            for bias in [0,0.0001,0.0005,0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
                  x_q = self.quantize(x_clone,delta,taus,bias)
                  score = lp_loss(x_clone, x_q, p=2, reduction='all')
                  if score < best_score:
                     best_score = score
                     cur_bais = bias
                     new_taus= taus
                     self.delta1 = delta
        print(self.delta1)
        print(new_taus)
        print(cur_bais)
        return cur_bais, new_taus


    def quantize(self, x, delta,taus,bias):
        x1 = x + bias
        x_int = round_ste(-1 * (x1 / delta).log2() * taus)
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        n = torch.floor(-1 * (x_quant / taus))
        m = (-x_quant % taus) / taus
        x_float_q = delta * (2 ** n) * 2 ** m
        x_float_q = x_float_q -bias
        return x_float_q
