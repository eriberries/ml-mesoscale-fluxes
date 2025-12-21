import numpy as np
import torch
import random

seed = 42
torch.manual_seed(seed)
if device == "cuda": torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# For the Dataloader 
def seed_worker(worker_id): # the worker_init_fn to specify in DataLoader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator() #the generator to specify in DataLoader
g.manual_seed(seed)
