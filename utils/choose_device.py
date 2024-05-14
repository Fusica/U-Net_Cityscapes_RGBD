import torch


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device('mps')
    else:
        print("Using CPU")
        return torch.device('cpu')
