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
        if not isinstance(p_list, list):
            raise TypeError('Probability is not a list!')
        r_all = 0
        for _, (alpha,p) in enumerate(zip(self.alpha_list, p_list)):
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
        x_norm = torch.div(x,m)
        return x_norm


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
    

class SparseSoftMax(object):
    def __init__(self, beta, dim):
        self.beta = beta
        self.dim = dim
    def weight(self, x, w):
        dim = self.dim
        beta = self.beta
        r_w = x*w
        mx = torch.abs(torch.max(r_w,dim=dim,keepdim=True)[0])
        if torch.any(mx==0):
            mx[mx==0] = 1
        r_norm = torch.div(r_w,mx) #rescale r
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
            predicate = self.a*x[:,:,dim] - self.b
        return predicate
    

class LogicalOperator(object):
    def __init__(self, oper, dim, avm=True, beta=False, tvar=False):
        '''
        Specify the type of logical operator
        To use variable-based operator, the input value 'tvar' is needed. Defalt: False
        If avm=True, then the averaged max is used, otherwise, the sparse softmax is used. Default: True
        To use sparse softmax, the input value 'beta' is needed. Defalt: False
        '''
        self.operation = oper
        if self.operation == 'logical': # variable-based logical operator
            if tvar is False:
                raise ValueError('Missing variable for variable-based logical operator!')
            self.tvar = tvar
        elif self.operation == 'and':
            self.tvar = -1
        elif self.operation == 'or':
            self.tvar = 1
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
        if self.operation == 'logical':
            tvar = Clip.apply(self.tvar)
        else:
            tvar = self.tvar
        if self.avm == False:
            w = STEstimator.apply(w)
        xx = x*tvar
        xr = self.max_function.forward(xx,w,keepdim)
        r = xr*tvar
        return r
    

class TemporalOperator(object):
    def __init__(self, oper, tau, t1, t2, avm=True, beta=False, tvar=False):
        '''
        Specify the type of temporal operator
        To use variable-based operator, the input value 'tvar' is needed.
        If avm=True, then the averaged max is used, otherwise, the sparse softmax is used.
        To use sparse softmax, the input value 'beta' is needed.
        '''
        self.operation = oper
        if self.operation == 'temporal': # variable-based logical operator
            if tvar is False:
                raise ValueError('Missing variable for variable-based temporal operator!')
            self.tvar = tvar
        elif self.operation == 'G': # always
            self.tvar = -1
        elif self.operation == 'F': # eventually
            self.tvar = 1
        else:
            raise ValueError("Temporal operation type is invalid!")
        if avm==True:
            self.max_function = AveragedMax(dim=1)
        else:
            if beta is False:
                raise ValueError('Missing beta for sparse softmax function!')
            self.max_function = SparseSoftMax(beta=beta,dim=1)
        self.time_weight = TimeFunction(tau,t1,t2)
    def forward(self, x, padding=False):
        '''
        x is of size [batch_size, T] where T is the length of signal.
        '''
        if len(x.shape)!=2:
            raise ValueError('Input dimension is invalid!')
        if self.operation == 'temporal':
            tvar = Clip.apply(self.tvar)
        else:
            tvar = self.tvar
        xx = x*tvar
        length = x.shape[-1]
        w = torch.tensor(range(length), requires_grad=False)
        wt = self.time_weight.forward(w)
        if padding is False:
            xr = self.max_function.forward(xx,wt,keepdim=False)
        else:
            rho_min = torch.min(xx,dim=1)[0]
            rho_pad = torch.unsqueeze(rho_min,1).repeat((1,length-1))
            x_pad = torch.cat((xx,rho_pad),dim=1)
            xr = torch.empty(x.shape)
            for i in range(length):
                xi = torch.clone(x_pad[:,i:(i+length)])
                ri = self.max_function.forward(xi,wt,keepdim=False)
                xr[:,i] = ri
        r = xr*tvar
        return r
    

class STLNeuralnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stlnn = nn.Sequential(
            nn.Predicate(),
            nn.Temporal(),
            nn.Temporal(),
            nn.Logical(),
            nn.Logical(),
        )