
def get_nelement(*args):
    return sum([sum([t.numel() for t in module.parameters()]) for module in args])