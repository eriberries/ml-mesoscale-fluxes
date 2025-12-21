import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
import threadpoolctl
threadpoolctl.threadpool_limits(limits=16)

# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# optuna hp exploration is not done using cross validation, cross validation is done for further finetuning

discard10 = True
cycle_step_int = 4
mode = "triangular"
loss_name = 
n_dropout = 1 

ismemory = False
print("ismemory: ", ismemory)

size_data = {}
train_dataset, val_dataset = Get_datasets(
    normalize=normalize_in, 
    normalize_output=normalize_out,
    print_shape = True, 
    which_fold=(1,1,0), 
    iscrossval=False, 
    k=None, 
    discard10=discard10, 
    ismemory=ismemory, 
    size_data=size_data
)

######################################################################################
# Hyper parameters specifications and reanges
######################################################################################
n_layers_min = 6
n_layers_max = 12

#lr_min = 1e-7
#lr_max = 1e-1

wd_min = 1e-7
wd_max = 1e-3


# L1s_min = 1e-4
# L1s_max = 1e-1

base_lr_min = 1e-6
base_lr_max = 3e-6
max_lr_min = None
max_lr_max = None


dropout_l_min = 0.005
dropout_l_max = 0.03

timeout = 14*60*60
n_trials = 100

######################################################################################

def define_model(
    trial,
    in_features = in_features, 
    out_features = out_features
):
    
    layers = []
    in_length=in_features
    
    for i in range(n_layers):

        if i < n_layers//4:
            out_length_factor = trial.suggest_int("n_units_l{}".format(i), 7, 14+1)
        elif i<n_layers//2:
            out_length_factor = trial.suggest_int("n_units_l{}".format(i), 4, 7+1)
        else: 
            out_length_factor = trial.suggest_int("n_units_l{}".format(i), 1, 4+1)
        
        out_length = 128*out_length_factor
        print("out_length: ", out_length)
        
        if i == 0: 
            layers.append(nn.Linear(in_length, out_length))
        if i > 0: 
            layers.append(nn.Linear(in_length, out_length, bias=False)) 

        activation = trial.suggest_categorical("activation", ["relu", "gelu"])

        if activation == "relu": 
            layers.append(nn.ReLU())
        if activation == "gelu": 
            layers.append(nn.GELU())
            
        layers.append(nn.BatchNorm1d(out_length))
        
        # if i < n_layers-1:
        if activation == "relu": 
            if i < n_dropout:
                p = trial.suggest_float(
                    "dropout_l{}".format(i), dropout_l_min, dropout_l_max
                )
                print("p: ", p)
                layers.append(nn.Dropout(p))
        
        in_length = out_length

    if mask_14Qlevels: 
        layers.append(MaskedLinear(in_length, out_features, mask))
    else: 
        layers.append(nn.Linear(in_length, out_features))
    
    return nn.Sequential(*layers)


torch.manual_seed(seed)
if device == "cuda": 
    torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# For the Dataloader 
def seed_worker(worker_id): # the worker_init_fn to specify in DataLoader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator() #the generator to specify in DataLoader
g.manual_seed(seed)


def objective(trial):
    "The objective is the MSE loss, which is to be minimized"
    
    # ismemory = trial.suggest_categorical("ismemory", [True, False])
    # print("ismemory: ", ismemory)
    # size_data = {}
    # train_dataset, val_dataset = Get_datasets(
    #     normalize=normalize_in, 
    #     normalize_output=normalize_out,
    #     print_shape = True, 
    #     which_fold=(1,1,0), 
    #     iscrossval=False, 
    #     k=None, 
    #     discard10=discard10, 
    #     ismemory=ismemory, 
    #     size_data=size_data
    # )

    cyclestep = size_data["train_X.shape"][0]/1024
    cyclestep = int(cyclestep) + (cyclestep != int(cyclestep))
    try:
        step_size_up = step_size_up 
    except NameError:
        step_size_up = cycle_step_int*cyclestep
    
    in_features = size_data["train_X.shape"][1]
    out_features = size_data["train_y.shape"][1]

    model = define_model(trial, in_features, out_features).to(device)
    
    wd = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    print("weight_decay: ", wd)

    # wd = 0
    # l1_reg_strength = trial.suggest_float("L1reg_strength", L1s_min, L1s_max, log=True)

    base_lr = trial.suggest_float("base_lr", base_lr_min, base_lr_max, log=True)
    print("base_lr: ", base_lr)

    lr_step = trial.suggest_float("lr_step", 1e2, 4e2, log=True)
    print("lr_step: ", lr_step)

    max_lr = base_lr * lr_step
    lr = base_lr

    # mode = trial.suggest_categorical("cyclic_mode", ["triangular", "triangular2", "exp_range"])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr, max_lr, step_size_up=step_size_up, 
        mode=mode, cycle_momentum = True
    ) 
    
    # loss_name = trial.suggest_categorical("loss_name", ["MSLE", "BalancedL1Loss", "Huberloss"]) # and more
    criterion = pick_loss(loss_name)

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

    for epoch in range(epochs):
        model.train()
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss = torch.mean(loss, dim=0)
            # loss += l1_regularization(model, l1_reg_strength)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        loss_val = 0
        with torch.no_grad():
            for batch_idx, (X_val, y_val) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)

                loss = torch.nn.MSELoss()(y_pred, y_val) 
                loss_val += loss.item()
                
        loss_val_average = loss_val/len(val_loader)
        print(epoch, loss_val_average)
        
        trial.report(loss_val_average, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss_val_average


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials, timeout=timeout)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("\t {} = {}".format(key, value))

print(f"External hyperparameters = {n_layers, n_dropout, normalize_in, normalize_out, mode, cycle_step_int}")
