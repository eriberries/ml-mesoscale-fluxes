import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import threadpoolctl
threadpoolctl.threadpool_limits(limits=16)

#from config import epochs, device, batchsize, g, num_workers, seed_worker, print_shape
#from prepare_dataset import Get_datasets


def define_model():
    layers = []
    layers.append(nn.Linear(244, 88))
    return nn.Sequential(*layers)


size_data = {}
r_c = (False, False)
train_dataset, val_dataset = Get_datasets(
    normalize_input="level", 
    normalize_output="momentum",
    print_shape = True, 
    which_fold=(1,1,0),
    isfinal=False, 
    size_data=size_data,
    ismemory=False, 
    which_out_var = "all", 
    isremoveextr=False, 
    whichMoisture= "Q",
    r_c = r_c
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batchsize, 
    num_workers=num_workers,
    shuffle=True, 
    worker_init_fn=seed_worker, 
    generator=g
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batchsize, 
    shuffle=False, 
    num_workers=num_workers
)

model = define_model().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(16):
    model.train()
    loss_train = 0
    for i, (X, y) in enumerate(train_loader):
        
        X, y = X.to(device).float(), y.to(device).float()
        y_pred = model.forward(X)
        loss = criterion(y_pred, y)
        loss_train += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    loss_train_eval = 0
    loss_val = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device).float(), y.to(device).float()
            y_pred = model.forward(X)
            loss = criterion(y_pred, y)
            loss_train_eval += loss.item()
            
        for i, (X, y) in enumerate(val_loader):
            X, y = X.to(device).float(), y.to(device).float()
            y_pred = model.forward(X)
            loss = criterion(y_pred, y)
            loss_val += loss.item()
    
    loss_train_average = loss_train / len(train_loader)
    loss_train_eval_average = loss_train_eval / len(train_loader)
    loss_val_average = loss_val / len(val_loader)

    print(f'''
    Epoch = {epoch}
    loss train = {loss_train_average}
    loss train (eval) = {loss_train_eval_average}
    loss val = {loss_val_average}
    ''')

torch.save(model, "/home/eismaili/Masters_Thesis" +  f"/Saved_models/MLR_ANN_epoch{epoch+1}.pt")
