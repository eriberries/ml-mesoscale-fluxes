import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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

from Model_config.CV_config import config_list

for conf in config_list:

    # load the variables from the list of hyperparameters 
    n_layers = conf["n_layers"]
    if "n_dropout" in conf.keys(): 
        n_dropout = conf["n_dropout"]
    out_length_list = conf["specify_nodes"]
    if "specify_dp" in conf.keys():  
        dropout_list = conf["specify_dp"]
    weight_decay = conf["weight_decay"]
    base_lr = conf["base_lr"]
    lr_factor = conf["lr_factor"]
    cycle_step_int = conf["cycle_step_int"]
    normalize = conf["norm_in"]
    normalize_output = conf["norm_out"]
    max_lr = base_lr * lr_factor
    lr = base_lr
    cyclic_mode = conf["cyclic_mode"]
    criterion = conf["criterion"]
    ismemory = conf["ismemory"]
    if "step_size_up" in conf.keys(): 
        step_size_up = conf["step_size_up"]
    activation = conf["activation"]
    
    if "epochs" in conf.keys(): 
        epochs = conf["epochs"]
    if "moisture" in conf.keys(): 
        whichMoisture = conf["moisture"]
    else: 
        whichMoisture = "Q"
    if "optimizer_name" in conf.keys(): 
        optimizer_name = conf["optimizer_name"]
    count_conf += 1

    
    # hyper parameter text to be printed
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
        cycle_step_int = {cycle_step_int}
        isauxilary: {isauxiliary}
    '''
    print(hp_text)

    # model
    def define_model(
        in_features= in_features, 
        out_features = out_features
    ):
        layers = []
    
        in_length=in_features
        
        for i in range(n_layers):
            out_length = int(out_length_list[i])
            
            if i == 0: layers.append(nn.Linear(in_length, out_length))
            if i > 0: layers.append(nn.Linear(in_length, out_length, bias=False)) 
    
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
        if mask_14Qlevels: layers.append(MaskedLinear(in_length, out_features+add_to_out, mask))
        else: layers.append(nn.Linear(in_length, out_features+add_to_out))
    
        return nn.Sequential(*layers)
    
    k_loss_list = []
    r2_k_list = []
    
    for k in range(K): 
        print("Fold #: ", k, f" / {K}")
        
        size_data = {}
        means_out = {}
        train_dataset, val_dataset = Get_datasets(
            normalize=normalize, 
            normalize_output=normalize_output,
            print_shape = print_shape, 
            which_fold=(1,1,0),
            iscrossval=iscrossval, 
            k=k, 
            size_data=size_data,
            means_out = means_out,
            ismemory=ismemory, 
            which_out_var = which_out_var, 
            isremoveextr=isremoveextr, 
            whichMoisture= whichMoisture
        )
        means_out = means_out["mean"]
        
        
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
        
        test_loader = None
        cyclestep = size_data["train_X.shape"][0]/1024
        cyclestep = int(cyclestep) + (cyclestep != int(cyclestep))
        
        try:
            step_size_up = step_size_up 
        except NameError:
            step_size_up = cycle_step_int*cyclestep
            
        print("step_size_up: ", step_size_up)
        
        model = define_model(
            in_features= size_data["train_X.shape"][1],
            out_features= size_data["train_y.shape"][1]
        ).to(device)
        
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(),
            lr=lr, weight_decay=weight_decay
        )
    
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr, max_lr, mode=cyclic_mode, 
            step_size_up=step_size_up, cycle_momentum = cycle_momentum
        ) 
        scheduler2 = optim.lr_scheduler.CyclicLR(
            optimizer, 0.8*base_lr, 0.08*max_lr,
            mode=cyclic_mode, step_size_up=step_size_up,
            cycle_momentum = cycle_momentum
        ) 
        
        # scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2) 
        
        trainer = Trainer(
            model, optimizer, criterion, 
            scheduler=scheduler, scheduler2=scheduler2, scheduler3=None,
            L1reg=L1reg, l1_reg_strength = l1_reg_strength,
            epoch_start=epoch_start, totime=totime, tosave=tosave,
            model_title=title
        )

        eval_list = []
        count_toolarge = 0
        count_conslarge = 0
        isbreak = False
        
        for epoch in range(epochs):
            if totime: 
                start_t = time.time()
                
            trainer.train_model(train_loader)
            
            if totime: 
                intermediate_t = time.time()
                
            computeR2 = False if epoch < epochs-1 else True
            
            trainer.eval_model(
                train_loader, val_loader, test_loader, 
                istrainloss=False, istestloss=False,
                computeR2=computeR2, means_out=means_out
            )

            if totime: final_t = time.time()
    
            validation_loss = trainer.losses_val_list[-1]
            training_loss = trainer.losses_train_list[-1]
            standard_training_loss = trainer.standard_losses_train_list[-1]
            eval_list.append(validation_loss)


            # break loop if evaluation loss is too high for consecutive steps
            # as well as the loop for the other folds 
            # if validation_loss > eval_list[0] and eval_list[0] <0.19:#< 0.17:
            #     count_conslarge +=1
            # elif validation_loss > 0.17: #0.17:
            #     count_conslarge +=1
            # else:
            #     count_conslarge = 0
            # if validation_loss > 1:
            #     count_toolarge += 1
            # if count_toolarge == 5 or count_conslarge == 4:
            #     isbreak = True
            # if isbreak: break

            print(f'''Epoch {epoch+ epoch_start}: 
            \t Training Loss = {training_loss} \t Standard Training Loss (eval) = {standard_training_loss}
            \t Validation Loss (eval) = {validation_loss}
            ''')
            
            print("Learning rate: ", trainer.learningrate)
            if totime: 
                print(f"time evaluation loop : {final_t-intermediate_t}, time training loop: {intermediate_t-start_t} seconds")
                
        # if isbreak: 
        #     with open("Results_crossvalidation.txt", 'a') as tgt:
        #         tgt.write(f'''Training/evaluation loop stopped, because of bad evaluation results \n for following Hyper parameters: ''')
        #         tgt.write(f"last validation loss: {validation_loss} at epoch {epoch}")
        #         tgt.write(hp_text)
        #         tgt.write(f'''step_size_up = {step_size_up}''')
        #         tgt.write('\n \n \n')
        #     break
    
        k_loss_list.append(validation_loss)
        torch.save(trainer.model, "/home/eismaili/2_Thesis/Scripts/NN" +  f"/Saved_models/{title}_k{k}.pt")
        with open("CV_loss_curves.txt", 'a') as tgt:
            if k== 0:
                tgt.write(hp_text) 
            tgt.write(f"k: {k} / {K}")
            tgt.write('\n')
            tgt.write(f"train losses: {trainer.losses_train_list} \n")
            tgt.write(f"val losses:   {trainer.losses_val_list} \n")
            tgt.write(f"r2: {trainer.r2_val_list}")
            tgt.write('\n \n \n')

    
    if not isbreak:
        print()
        print(f'''RESULT: average validation loss: {sum(k_loss_list) / len(k_loss_list)}''')
        with open("Results_crossvalidation.txt", 'a') as tgt:
            tgt.write(f'''RESULT: average validation loss: {sum(k_loss_list) / len(k_loss_list)} \n for following Hyper parameters: ''')
            tgt.write(hp_text)
            tgt.write(f'''step_size_up = {step_size_up}''')
            tgt.write('\n \n \n')

