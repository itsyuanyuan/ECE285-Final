class Woptim:
    def __init__(self, d_size, warmup, optim):
        self.optimizer = optim
        self.d_size = d_size
        self.warmup = warmup
        self._step = 0
        self._lr = 0
        
    def state_dict(self):
        sd = {}
        for key, value in self.__dict__.items():
            if key != 'optimizer':
               sd["key"] = value
        return sd
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    def step(self):
        self._step +=1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr
        self.optimizer.step()
    def rate(self, step = None):
        if step is None
            step = self._step
        return (self.d_size).pow(-0.5) * min(step.pow(-0.5), step * self.warmup.pow(-1.5))
        
