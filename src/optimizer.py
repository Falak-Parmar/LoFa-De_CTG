import math
import torch
from torch.optim import Optimizer

class PythonAdamW(Optimizer):
    """
    A Hybrid CPU-GPU implementation of the AdamW optimizer.
    Bypasses buggy PyTorch C++ MPS kernels on macOS by performing weight updates on the CPU
    while keeping the model and gradient computation on the GPU (MPS).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(PythonAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                device = p.device

                # Move parameters and gradients to CPU for fast stable updates
                p_cpu = p.to("cpu")
                grad_cpu = grad.to("cpu")

                state = self.state[p]

                # State initialization on CPU
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_cpu)
                    state['exp_avg_sq'] = torch.zeros_like(p_cpu)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Perform weight decay on CPU
                if weight_decay != 0:
                    p_cpu.mul_(1.0 - lr * weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1.0 - beta2)

                # Bias correction terms
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                # Compute step size
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p_cpu.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Copy updated weights back to GPU
                p.copy_(p_cpu.to(device))

        return loss
