# Device / runtime
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
num_workers = 6
batchsize = 1024
epochs = 30


