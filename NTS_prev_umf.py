import torch
from torch import detach
import math as m
from torch.optim.optimizer import Optimizer
from torch.autograd import backward
from copy import deepcopy
import random
import time
from fgm import FGM

class NTS(Optimizer):
    """Implements Nesterov Three squares method.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        function : some fuction for optimization or object of neural network
        x_in : vector of input data for neural network
        y_target : target vector for neural network
        lr (float, optional): learning rate (default: 1e-2)
        L (float, optional or adaptive) : Lipshitz constant
        epoch (int) : quantity iterates for subtask in Nesterov Three Squares method
        adaptive_lr (bool, optional) : learning rate adaptivity
        adaptive_L (bool, optional) : Lipshitz adaptivity
        limit_L (float, optional) : lower limit for Lipshitz constant
        limit_recurse(int, optional) : maximum number of decrease of Lipshitz constant
        r (int, optional) : sample size of functions is optimized
        seed (int, optional) : random seed
        break_L (float, optional) : upper limit of Lipshitz constant for stop of optimization
        break_lr (float, optional) : low limit for learning rate for stop of optimization
        subsolver (optional) : optimization method for solving subtask

    """

    def __init__(self, params, function, grad_function, subsolver=FGM, x_in=None, y_target=None, lr=1e-2, L=1, \
        epoch=100, adaptive_lr=False, adaptive_L=False, limit_L=1e-10, \
            limit_recurse=10, r=None, seed=None, break_L=1e20, break_lr=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        random.seed(seed)
        
        defaults = dict(function=function, grad_function=grad_function, subsolver=subsolver, x_in=x_in, y_targer=y_target, lr=lr, L=L, epoch=epoch, 
                        adaptive_lr=adaptive_lr, adaptive_L=adaptive_L, limit_L=limit_L, 
                        limit_recurse=limit_recurse, n_recurse=0, r=r, flag_L=False, flag_lr=False,
                        indices=None, break_L=break_L, break_lr=break_lr, flag_c = True)
        super(NTS, self).__init__(params, defaults)
  

                
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.defaults['L'] >= self.defaults['break_L'] \
            or self.defaults['lr'] <= self.defaults['break_lr']:
            return loss

        if self.defaults['flag_c']:
            self.set_indices()
            params = self.get_params(self.param_groups)
            with torch.no_grad():
                self.defaults['F'] = self.defaults['function'](*params)[self.defaults['indices']]
                self.defaults['grad_F'] =self.defaults['grad_function']\
                    (*params)[self.defaults['indices']]
            #print(self.defaults['F'])
            #print(self.defaults['grad_F'])

        param_groups_k = deepcopy(self.param_groups)
        params_k = self.get_params(param_groups_k)
        self.make_grad(False)

        optimizer = self.defaults['subsolver'](params_k, lr=self.defaults['lr'])

        params[0].requires_grad_(False)
        for i in range(self.defaults['epoch']):

            start_time = time.time()
            #print(i)
            #print(params_k)
            
            f_nes = self.get_f_nes(params, params_k)

            optimizer.zero_grad()
            f_nes.backward()
            optimizer.step()

            #print('suboptim step = {} its time = {}'.format(i, time.time() - start_time))

        self.make_grad()
                
        f_nes = self.get_f_nes(params, params_k)

        buf_params = deepcopy(self.param_groups)
        self.copy_params(param_groups_k)

        f = (self.defaults['F'] ** 2).sum() ** 0.5
        f_i = self.get_f_value()

        #print('f(xk + 1) = {}  f(xk) = {}  f_nes = {}'.format(f_i, f, f_nes))
        #print('L = {} lr = {}'.format(self.defaults['L'], self.defaults['lr']))
        #if f_nes > 100 and f_i < f_nes:
            #print(params_k)
            #print('f(xk + 1) = {}  f(xk) = {}  f_nes = {}'.format(f_i, f, f_nes))
        #if f_i > f_nes:
        #    print('f(xk + 1) = {}  f(xk) = {}  f_nes = {}'.format(f_i, f, f_nes))
        #print('f_i = {}  f = {}  L = {}  lr = {}  n_recurse = {}'.format(f_i, f, self.defaults['L'], self.defaults['lr'], self.defaults['n_recurse']))
        #print(self.defaults['indices'])
        #print(self.param_groups[0]['params'])
        #q = input()
        if self.defaults['adaptive_lr']:
            if torch.isnan(f_i) or torch.isinf(f_i):
                
                self.defaults['flag_lr'] = True
                self.defaults['lr'] *= 1e-1
                self.copy_params(buf_params)
                return self.step(closure=closure)
                
            else:

                if self.defaults['lr'] < 1e-1 and not self.defaults['flag_lr'] and f_i <= f:
                    #self.defaults['flag_lr'] = False
                    self.defaults['lr'] *= 1e1

            if f_i > f:
                #print('!as')
                self.defaults['lr'] *= 1e-1
                self.defaults['flag_lr'] = True
        else:
            if torch.isnan(f_i):
                raise Exception('Loss function value nan. Try choose smaller value of learning rate')
            if torch.isinf(f_i):
                raise Exception('Loss function value inf. Try choose smaller value of learning rate')

        if self.defaults['adaptive_L']:
            if f_i <= f:
                
                if self.defaults['L'] > self.defaults['limit_L'] and self.defaults['n_recurse'] < self.defaults['limit_recurse']:
                    self.defaults['L'] /= 2
                else:
                    self.defaults['flag_L'] = False
                    self.defaults['flag_lr'] = False
                    self.defaults['n_recurse'] = 0
                    self.defaults['flag_c'] = True
                    return loss
                
                if self.defaults['flag_L']:
                    self.defaults['flag_L'] = False
                    self.defaults['flag_lr'] = False
                    self.defaults['n_recurse'] = 0
                    self.defaults['flag_c'] = True
                    return loss
                else:
                    self.copy_params(buf_params)
                    self.defaults['n_recurse'] += 1
                    return self.step(closure=closure)
            else:
                self.defaults['flag_L'] = True
                self.defaults['L'] *= 2
                self.copy_params(buf_params)
                return self.step(closure=closure)

        self.defaults['flag_c'] = True

        return loss


    def set_x(self, x_in, y_target):
        """Set input vector and target vector 

        Arguments:
            x_in : input vector
            y_target : target vector

        """

        self.defaults['x_in'] = x_in
        self.defaults['y_target'] = y_target
        

    def set_indices(self):

        if self.defaults['x_in'] is None:
            
            params = self.get_params(self.param_groups)
            F = self.defaults['function'](*params)
            n = F.shape[0]
        
        else:

            n = self.defaults['y_target'].shape[0]

        if self.defaults['r'] is None:
            self.defaults['indices'] = list(range(n))
        else:
            self.defaults['indices'] = random.sample(list(range(n)), self.defaults['r'])
        


    def func_and_grad(self):

        #print('Jacobian calculating is started')
        start_time = time.time()

        self.set_indices()

        if self.defaults['x_in'] is None:
            params = self.get_params(self.param_groups)
            #print('params = {}'.format(*params))
            #print('junc = {}'.format(self.defaults['indices']))
            F = self.defaults['function'](*params)[self.defaults['indices']]
        else:
            F = (self.defaults['function'](self.defaults['x_in']) - self.defaults['y_target'])[self.defaults['indices']].reshape(-1, 1)
            #print('x : {} F : {}'.format(self.defaults['x_in'].shape, F.shape))

        grad_F = []
        #print('F = {}'.format(F))
        size = F.shape[0]
        for i, F_i in enumerate(F):

            #print('{} of {}'.format(i, size))
            #print('F_i = {} done\n\n'.format(F_i))
            backward(F_i, retain_graph=True)

            grad_groups = []
            for group in self.param_groups:

                grad_group = []
                for p in group['params']:

                    grad_group.append(p.grad.data)
                    p.grad.data = torch.zeros_like(p.grad.data)
                
                grad_groups.append(grad_group)
            
            grad_F.append(grad_groups)

        F = detach(F)

        #print('Jacobian calculating time = {}'.format(time.time() - start_time))

        return F, grad_F
                
        
    def make_grad(self, value=True):

        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad_(value)
        return         


    def get_f_value(self):

        if self.defaults['x_in'] is None:
            
            params = self.get_params(self.param_groups)
            return (self.defaults['function'](*params)[self.defaults['indices']] ** 2).sum() ** 0.5

        else:

            return ((self.defaults['function'](self.defaults['x_in']) - self.defaults['y_target'])[self.defaults['indices']] ** 2).sum() ** 0.5


    def get_params(self, param_groups_k):

        params = []

        for group in param_groups_k:
            for p in group['params']:
                params.append(p)

        return params

    def copy_params(self, param_groups_c):

        for group, group_c in zip(self.param_groups, param_groups_c):
            for p, p_c in zip(group['params'], group_c['params']):
                with torch.no_grad():
                    p.copy_(p_c)


    def get_f_nes(self, params, params_k):
        
        start_time = time.time()
        #print('Subfunction calculating is started!')

        f_1 = self.defaults['F'] + self.defaults['grad_F'].mm((params_k[0] - params[0]).reshape(-1, 1))
        f_1 = (f_1 ** 2).sum()

        f_2 = self.defaults['L'] * ((params_k[0] - params[0]) ** 2).sum() / 2

        func = (self.defaults['F'].detach() ** 2).sum()

        f_nes = (func + f_1) / (2 * func ** 0.5) + f_2

        #print('Subfunction calculating time = {}'.format(time.time() - start_time))

        return f_nes