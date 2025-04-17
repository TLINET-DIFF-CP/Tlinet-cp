import torch
import torch.nn as nn

class STEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        # g -> gs
        g_clip = torch.clamp(g, min=0, max = 1)
        gs = g_clip.clone()
        gs[gs>=0.5] = 1
        gs[gs<0.5] = 0
        return gs
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class Clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()
        gs[gs>=0] = 1
        gs[gs<0] = -1
        return gs
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input
    

class Bimodal_reg(object):
    def __init__(self, alpha_list):
        if not isinstance(alpha_list, list):
            raise TypeError('Weight for probability regularizer is not a list!')
        self.alpha_list = alpha_list
    def get_reg(self, p_list):
        # if not isinstance(p_list, list):
        #     raise TypeError('Probability is not a list!')
        if len(self.alpha_list) != len(p_list):
            raise TypeError('Please specify weight for each layer!')
        r_all = 0
        for _, (alpha,p) in enumerate(zip(self.alpha_list, p_list)):
            if len(p.shape) == 1:
                n_row = p.shape[0]
                r = 0
                for i in range(n_row):
                    pi = p[i]
                    r += pi*(1-pi)
                r_all += alpha*(r)
            else:
                n_row = p.shape[0]
                n_col = p.shape[1]
                r = 0
                for i in range(n_row):
                    for j in range(n_col):
                        pi = p[i,j]
                        r += pi*(1-pi)
                r_all += alpha*(r)
        return r_all


class Batch_Normalization(object):
    def __init__(self, dim):
        self.d = dim
    def forward(self,x):
        mu = torch.mean(x, self.d, keepdim=True)
        var = torch.var(x, self.d, keepdim=True)
        x_norm = torch.div(x-mu,torch.sqrt(var))
        return x_norm
    

class Normalization_max(object):
    def __init__(self, dim):
        self.d = dim
    def forward(self,x):
        m = torch.max(torch.abs(x),dim=self.d,keepdim=True)[0]
        m0 = torch.max(m,dim=0,keepdim=True)[0]
        x_norm = torch.div(x,m0)
        return x_norm, m0


class TimeFunction(object):
    def __init__(self, tau, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.tau = tau
        self.relu = torch.nn.ReLU()
    def forward(self, w):
        f1 = (self.relu(w-self.t1+self.tau)-self.relu(w-self.t1))/self.tau
        f2 = (self.relu(-w+self.t2+self.tau)-self.relu(-w+self.t2))/self.tau
        w = torch.min(f1,f2)
        return w


class TimeFunction(object):
    def __init__(self):
        pass
    def forward(self, w, tau, t1, t2):
        relu = torch.nn.ReLU()
        # relu = torch.nn.Sigmoid()
        f1 = (relu(w-t1+tau)-relu(w-t1))/tau
        f2 = (relu(-w+t2+tau)-relu(-w+t2))/tau
        w = torch.min(f1,f2)
        return w


class TimeFunction_revise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, tau, t1, t2):
        relu = torch.nn.ReLU()
        relu = torch.nn.Sigmoid()
        f1 = (relu(w-t1+tau)-relu(w-t1))/tau
        f2 = (relu(-w+t2+tau)-relu(-w+t2))/tau
        tw = torch.min(f1,f2)
        ctx.save_for_backward(w,tau,t1,t2,f1,f2)
        return tw
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        w,tau,t1,t2,f1,f2 = ctx.saved_tensors
        f11 = ((w-t1+tau)>0)*(-1)
        f12 = ((w-t1)>0)*(-1)
        f21 = ((-w+t2+tau)>0)*(1)
        f22 = ((-w+t2)>0)*(1)
        tw1_grad = (f1<=f2)*(f11-f12)/tau
        tw2_grad = (f2<=f1)*(f21-f22)/tau
        beta = 5
        # sigma_t1 = 1/(1+torch.exp(-beta*(w-t1+0.5)))
        # sigma_t2 = 1/(1+torch.exp(beta*(w-t2+0.5)))
        w_grad = torch.tensor(0,dtype=torch.float64)
        tau_grad = torch.tensor(0,dtype=torch.float64)
        ts1_grad = (-beta*torch.exp(-beta*(w-t1+0.5)))/torch.square(1+torch.exp(-beta*(w-t1+0.5)))
        ts2_grad = (beta*torch.exp(beta*(w-t2-0.5)))/torch.square(1+torch.exp(beta*(w-t2-0.5)))
        t1_grad_temp = tw1_grad*grad_input
        t2_grad_temp = tw2_grad*grad_input
        # return w_grad, tau_grad, tw1_grad*grad_input, tw2_grad*grad_input
        # return w_grad, tau_grad, t1_grad*grad_input, t2_grad*grad_input
        if torch.sum(torch.abs(t1_grad_temp))<torch.sum(torch.abs(ts1_grad*grad_input)):
            t1_grad = ts1_grad*grad_input
        else:
            t1_grad = t1_grad_temp
        if torch.sum(torch.abs(t2_grad_temp))<torch.sum(torch.abs(ts1_grad*grad_input)):
            t2_grad = ts2_grad*grad_input
        else:
            t2_grad = t2_grad_temp
        return w_grad, tau_grad, t1_grad, t2_grad
    

class SparseSoftMax(object):
    def __init__(self, beta, h, dim):
        self.beta = beta
        self.h = h
        self.dim = dim
    def weight(self, x, w):
        dim = self.dim
        beta = self.beta
        # print(x.get_device(),w.get_device())
        r_w = x*w
        mx = torch.abs(torch.max(r_w,dim=dim,keepdim=True)[0])
        if torch.any(mx==0):
            mx[mx==0] = 1
        r_norm = torch.div(self.h*r_w,mx) #rescale r
        r_exp = torch.exp(beta * r_norm)
        r_sum = torch.sum(r_exp,dim=dim,keepdim=True)
        s_norm = torch.div(r_exp,r_sum)
        return s_norm
    def forward(self, x, w, keepdim=False):
        '''
        x: input of size [batch_size, ...] or [batch_size, T, dim]
        w is a one-dimensional vector. 
        '''
        dim = self.dim
        if len(w.shape)>1:
            raise ValueError('Dimension of weight is invalid!')
        if dim != len(x.shape)-1:
            raise ValueError('Invalid operation! Please check!')
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum
        s_norm = self.weight(x, w_norm)
        sw = torch.mul(s_norm,w_norm)
        denominator = torch.sum(sw,dim=dim,keepdim=keepdim)
        numerator = torch.mul(sw, x)
        numerator = torch.sum(numerator,dim=dim,keepdim=keepdim)
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        rho = numerator/denominator
        return rho
        

class AveragedMax(object):
    def __init__(self, dim):
        self.dim = dim
    def prob(self, p):
        dim = self.dim
        if dim==2:
            prob = torch.empty(p.shape)
            for i in range(p.shape[dim]):
                pi = 1
                for j in range(i+1):
                    if j==i:
                        pi = pi*p[:,:,j]
                    else:
                        pi = pi*(1-p[:,:,j])
                prob[:,:,i] = pi
        if dim==1:
            prob = torch.empty(p.shape)
            for i in range(p.shape[dim]):
                pi = 1
                for j in range(i+1):
                    if j==i:
                        pi = pi*p[:,j]
                    else:
                        pi = pi*(1-p[:,j])
                prob[:,i] = pi
        return prob
    def forward(self, x, p, keepdim=False):
        '''
        x: input of size [batch_size, ...] or [batch_size, T, dim]
        p is a one-dimensional vector. 
        '''
        dim = self.dim
        if dim != len(x.shape)-1:
            raise ValueError('Invalid operation! Dimension mismatch!')
        xs, pindex = torch.sort(x,dim=dim,descending=True)
        psort = p[pindex]
        pw = self.prob(psort)
        expectation = torch.sum(xs*pw,dim=dim,keepdim=keepdim)
        return expectation
    

class Predicate(object):
    def __init__(self, a, b, dim=False):
        '''
        dim is specified if predicates is computed along that dimension. Default: False
        b is a 1-d scalar
        '''
        self.a = a
        self.b = b
        self.dim = dim
        if dim:
            if not isinstance(dim,int):
                raise TypeError('Dimension needs to be an integer!')
    def forward(self, x):
        '''
        x is of size [batch_size, T, dim] where T is the length of signal.
        output is of size [batch_size, T].
        '''
        dim = self.dim
        if dim is False:
            predicate = torch.matmul(x,self.a) - self.b
        else:
            # predicate = self.a*x[:,:,0] + x[:,:,1] - self.b
            predicate = self.a*x[:,:,dim] - self.b
        return predicate
    

class LogicalOperator(object):
    def __init__(self, oper, dim, avm=True, beta=False, type_var=False):
        '''
        Specify the type of logical operator
        To learn the type of operator, the input value 'type_var' is needed.
        If avm=True, then the averaged max is used, otherwise, the sparse softmax is used. Default: True
        To use sparse softmax, the input value 'beta' is needed. Defalt: False
        '''
        self.operation = oper
        if self.operation == 'logical': # variable-based logical operator
            if type_var is False:
                raise ValueError('Missing variable for variable-based logical operator!')
            self.type_var = type_var
        elif self.operation == 'and':
            self.type_var = torch.tensor(0,dtype=torch.float64,requires_grad=False)
        elif self.operation == 'or':
            self.type_var = torch.tensor(1,dtype=torch.float64,requires_grad=False)
        else:
            raise ValueError("Logical operation type is invalid!")
        self.avm = avm
        if avm==True:
            self.max_function = AveragedMax(dim=dim)
        else:
            if beta is False:
                raise ValueError('Missing beta for sparse softmax function!')
            self.max_function = SparseSoftMax(beta=beta,dim=dim)
    def forward(self, x, w, keepdim=False):
        if self.avm == False:
            w = STEstimator.apply(w)
        xmin = (-1)*torch.clone(x)
        xmax = torch.clone(x)
        xrmin = self.max_function.forward(xmin,w,keepdim)
        xrmax = self.max_function.forward(xmax,w,keepdim)
        r = self.type_var*xrmax + (1-self.type_var)*(-1)*xrmin
        return r
    

class TemporalOperator(object):
    def __init__(self, oper, tau, t1, t2, beta=False, h=False, type_var=False):
        '''
        Specify the type of temporal operator
        To learn the type of operator, the input value 'type_var' is needed.
        To use sparse softmax, the input value 'beta' is needed.
        '''
        self.operation = oper
        if self.operation == 'temporal': # learn type of temporal operator
            if type_var is False:
                raise ValueError('Missing variable for learning temporal operator!')
            self.type_var = type_var
        elif self.operation == 'G': # always
            self.type_var = torch.tensor(0,dtype=torch.float64,requires_grad=False)
        elif self.operation == 'F': # eventually
            self.type_var = torch.tensor(1,dtype=torch.float64,requires_grad=False)
        else:
            raise ValueError("Temporal operation type is invalid!")
        if beta is False:
            raise ValueError('Missing beta for sparse softmax function!')
        else:
            self.max_function = SparseSoftMax(beta=beta,h=h,dim=1)
        self.tau = tau
        self.t1 = t1
        self.t2 = t2
        self.time_weight = TimeFunction()
        # self.time_weight = TimeFunction_revise.apply
    def padding(self, x):
        length = x.shape[1]
        rho_min = torch.min(x,dim=1)[0]
        rho_pad = torch.unsqueeze(rho_min,1).repeat((1,length-1))
        x_pad = torch.cat((x,rho_pad),dim=1)
        return x_pad
    def forward(self, x, padding=False):
        '''
        x is of size [batch_size, T] where T is the length of signal.
        '''
        if len(x.shape)!=2:
            raise ValueError('Input dimension is invalid!')
        xmin = (-1)*torch.clone(x)
        xmax = torch.clone(x)
        length = x.shape[-1]
        w = torch.tensor(range(length), requires_grad=False)
        wt = self.time_weight.forward(w,self.tau,self.t1,self.t2)
        # wt = self.time_weight(w,self.tau,self.t1,self.t2)
        if padding is False:
            xrmin = self.max_function.forward(xmin,wt,keepdim=False)
            xrmax = self.max_function.forward(xmax,wt,keepdim=False)
        else:
            x_padmin = self.padding(xmin)
            x_padmax = self.padding(xmax)
            xrmin = torch.empty(x.shape)
            xrmax = torch.empty(x.shape)
            for i in range(length):
                xi = torch.clone(x_padmin[:,i:(i+length)])
                ri = self.max_function.forward(xi,wt,keepdim=False)
                xrmin[:,i] = ri
            for i in range(length):
                xi = torch.clone(x_padmax[:,i:(i+length)])
                ri = self.max_function.forward(xi,wt,keepdim=False)
                xrmax[:,i] = ri
        r = self.type_var*xrmax + (1-self.type_var)*(-1)*xrmin
        return r