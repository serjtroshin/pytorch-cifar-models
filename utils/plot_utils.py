import functools

class ConvergenceMeter(object):
    """Computes and stores current convergence stats"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_z = 0
        self.curr_diff = 0
        self.diffs = []
        self.bs = None

    def update(self, z):
        z = z.detach().cpu()
        self.curr_diff = (z - self.prev_z).norm().item()
        self.diffs.append(self.curr_diff)
        self.prev_z = z
        self.bs = z.shape[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val is None:
            self.val = None
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
        

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')
            
def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


class DEQMeter(object):
    def __init__(self, layer):
        self.grads = AverageMeter()
        self.forward_diffs = AverageMeter()
        self.backward_diffs = AverageMeter()
        self.pretrain_diffs = AverageMeter()
        self.layer = layer

    def update(self, input):
        self.grads.update(calc_grad_norm(self.layer), input.size(0))  
        self.forward_diffs.update(self.layer.info["forward_diffs"])
        self.backward_diffs.update(self.layer.info["backward_diffs"])
        self.pretrain_diffs.update(self.layer.info["pretrain_diffs"])