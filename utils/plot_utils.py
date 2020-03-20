

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
