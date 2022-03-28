from torch.optim import Optimizer


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay, *args, **kwargs):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
        pass

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p, c, ci in zip(group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                d_p.add_(c - ci)
                p.data.add_(d_p, alpha=-lr)
