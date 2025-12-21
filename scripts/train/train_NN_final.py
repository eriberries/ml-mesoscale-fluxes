import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import threadpoolctl
threadpoolctl.threadpool_limits(limits=16)

#from Functions_Classes_NN import Trainer, MaskedLinear
#from CostumParser import epochs, epoch_start, num_workers, isremoveextr, isauxiliary, which_out_var
#from LossFunctions import MSELoss_c, BalancedL1Loss, HuberLoss_c, L1Loss_c

idir = 
outdir = 


n_dropout = None
dropout_list = None

title = 
n_layers = 
n_dropout = 
out_length_list = 
dropout_list = 

normalize = 
normalize_output =

lr_factor = 
cycle_step_int = 
weight_decay = 
base_lr = 
max_lr = base_lr * lr_factor
lr = base_lr
cyclic_mode = 
criterion = 

ismemory = 

step_size_up = 

activation = 

epochs = 
whichMoisture =


def define_model(
    in_features = in_features, 
    out_features = out_features
):
    
    layers = []
    in_length=in_features
    
    for i in range(n_layers):
        out_length = out_length_list[i]
        
        if i == 0: 
            layers.append(nn.Linear(in_length, out_length))
        if i > 0: 
            layers.append(nn.Linear(in_length, out_length, bias=False)) 

        if activation == "relu": 
            layers.append(nn.ReLU())
        if activation == "gelu": 
            layers.append(nn.GELU())
            
        layers.append(nn.BatchNorm1d(out_length))
        
        # if i < n_layers-1:
        if activation == "relu": 
            if i < n_dropout:
                p = dropout_list[i]
                layers.append(nn.Dropout(p))
        
        in_length = out_length

    add_to_out = 0
    
    if isauxiliary: add_to_out +=1
        
    if mask_14Qlevels and (which_out_var == "OMEGAQ" or "all"): 
        
        mask = torch.ones(out_features, dtype=torch.float32)
        mask[:4] = 0
        
        layers.append(
            MaskedLinear(in_length, out_features+add_to_out, mask)
        )
    else: 
        layers.append(nn.Linear(in_length, out_features+add_to_out))

    return nn.Sequential(*layers)

size_data = {}
train_dataset, test_dataset = Get_datasets(
    normalize_input=normalize, 
    normalize_output=normalize_output,
    print_shape = print_shape, 
    which_fold=(1,0,1),
    isfinal=True, 
    size_data=size_data,
    ismemory=ismemory, 
    which_out_var = which_out_var, 
    isremoveextr=isremoveextr, 
    whichMoisture= whichMoisture,
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

val_loader = None
test_loader = DataLoader(
    test_dataset, 
    batch_size=batchsize, 
    shuffle=False, 
    num_workers=num_workers
)

cyclestep = size_data["train_X.shape"][0]/1024
cyclestep = int(cyclestep) + (cyclestep != int(cyclestep))

try:
    step_size_up = step_size_up 
except NameError:
    print("step_size_up does not exist, it has been calculated according to the sample length")
    step_size_up = cycle_step_int*cyclestep

hp_text = f'''

title: {title}
epochs = {epochs}
seed = {seed}

DATA
    normalize input: {normalize}
    normalize output: {normalize_output}
    ismemory: {ismemory}
    whichMoisture: {whichMoisture}

HIDDEN LAYERS
    Number of hidden layers: {n_layers}
    with values: {[out_length_list[i] for i in range(n_layers)]}
    Activation: {activation}

REGULARIZATION
    Number of dropout layers: {n_dropout}
    with values: {dropout_list}
    Weight decay (L2 regularization): {weight_decay}
    l1_reg_strength (L1 regularization): {l1_reg_strength}

OPTIMIZER, SCHEDULER AND CRITERION 
    optimizer = {optimizer_name}
    criterion = {criterion}
    base_lr = {base_lr}
    max_lr = base_lr * lr_factor, where lr_factor = {lr_factor}
    step_size_up = {step_size_up}
    isauxilary: {isauxiliary}
'''
print(hp_text)
print("num_out_features = ", size_data["train_y.shape"][1])


model = define_model(
    in_features= size_data["train_X.shape"][1],
    out_features= size_data["train_y.shape"][1]
).to(device)



optimizer = getattr(
    optim, optimizer_name
)(
    model.parameters(),
    lr=lr, 
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr, 
    max_lr,
    mode=cyclic_mode, 
    step_size_up=step_size_up,
    cycle_momentum = cycle_momentum
) 

scheduler2 = optim.lr_scheduler.CyclicLR(
    optimizer, 
    0.8*base_lr, 
    0.08*max_lr,
    mode=cyclic_mode, 
    step_size_up=step_size_up,
    cycle_momentum = cycle_momentum
) 

# scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2) 

trainer = Trainer(
    model, 
    optimizer, 
    criterion, 
    scheduler=scheduler, 
    scheduler2=scheduler2, 
    scheduler3=None,
    L1reg=L1reg, 
    l1_reg_strength = l1_reg_strength,
    epoch_start=epoch_start, 
    totime=totime, 
    tosave=True,
    model_title=title
)

trainer.Train_eval_loop(
    train_loader, 
    val_loader, 
    test_loader, 
    epochs=epochs, 
    istestloss=True, 
    isvalloss= False, 
    istrainloss=True, 
    isauxiliary = isauxiliary
)

print("Training losses:", trainer.losses_train_list)
print("Training MSE:", trainer.MSE_train_list)
print("Training MSE (eval):", trainer.MSE_train_eval_list)
print("Test MSE:", trainer.losses_test_list)

np.save('/home/eismaili/Masters_Thesis/Neural_Network/Losses/' + f"MSE_train_eval_{title}.npy", np.array(trainer.MSE_train_eval_list))
np.save('/home/eismaili/Masters_Thesis/Neural_Network/Losses/' + f"MSE_train_{title}.npy", np.array(trainer.MSE_train_list))
np.save('/home/eismaili/Masters_Thesis/Neural_Network/Losses/' + f"MSE_test_{title}.npy", np.array(trainer.losses_test_list))

