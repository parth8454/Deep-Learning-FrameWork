class SGD:
    def __init__(self,parameters,lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad  

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0 
        
        self.m = [0.0] * len(self.parameters)
        self.v = [0.0] * len(self.parameters) 

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad == 0.0:
                continue
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr*m_hat/(v_hat**0.5+self.eps)