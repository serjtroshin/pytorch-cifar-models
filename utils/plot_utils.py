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

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)