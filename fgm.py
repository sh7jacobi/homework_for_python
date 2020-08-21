import torch
import math as m
from torch.optim.optimizer import Optimizer

class FGM(Optimizer):
    """Implements Fast Gradient Method for convex functions.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)

    """

    def __init__(self, params, lr=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))


        defaults = dict(lr=lr)
        super(FGM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0.
                state['ak'] = 0.
                state['yk'] = p.data
  

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['yk'].share_memory_()
                state['ak'].share_memory_()
                
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                
                if p.grad.data.is_sparse:
                    raise RuntimeError("FGM is not compatible with sparse gradients")
                    
                ak = (m.sqrt(4 * state['ak']**2 + 1) + 1) / 2
                ak1 = (m.sqrt(4 * ak**2 + 1) + 1) / 2
                gamma = (1 - ak) / ak1
                yk1 = p.data - group['lr'] * grad
                p.data = (1 - gamma) * yk1 + gamma * state['yk']
                state['yk'] = yk1
                state['ak'] = ak
                
        return loss
