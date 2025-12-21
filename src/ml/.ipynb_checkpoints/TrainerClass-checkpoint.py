import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

#from config import epochs, device


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion,
        scheduler=None, 
        scheduler2=None, 
        scheduler3=None, 
        L1reg=False, 
        l1_reg_strength = 1e-4,
        epoch_start = 1,
        totime=False, 
        tosave=False, 
        model_title="some_model_name",
        toprintlr=True, 
        iscrossval=False
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
        self.L1reg = L1reg
        self.l1_reg_strength = l1_reg_strength

        self.epoch_start = epoch_start
        
        self.totime = totime
        self.tosave = tosave
        self.model_title = model_title

        self.losses_train_list = []
        self.standard_losses_train_list = []
        self.losses_val_list = []
        self.r2_val_list = []
        self.losses_test_list = []
        self.losses_train_eval_list = []
        self.MSE_train_list = []
        self.MSE_train_eval_list = []

        self.toprintlr = toprintlr
        self.learningrate = 0

        self.iscrossval = iscrossval
        self.epoch = 0
    
    def MSELoss_c(self, pred, true):
        loss = (pred-true)**2
        return torch.mean(loss, dim=1)

    def pickScheduler(self, epoch_th1=16, epoch_th2=24):
        
        if self.scheduler and not self.scheduler2 and not self.scheduler3:
            return self.scheduler
            
        elif self.scheduler and self.scheduler2 and not self.scheduler3:
            if self.epoch < epoch_th1:
                return self.scheduler
            else: return self.scheduler2
                
        elif self.scheduler and self.scheduler2 and self.scheduler3:
            if self.epoch < epoch_th1:
                return self.scheduler
            elif self.epoch < epoch_th2:
                return self.scheduler2
            else: return self.scheduler3
        
    
    def l1_regularization(self):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.sum(abs(param))
        return self.l1_reg_strength * l1_loss

    def train_model(
        self, 
        train_loader, 
        isauxiliary=False, 
        epoch_th1=16, 
        epoch_th2=24
    ):
        self.model.train()
        loss_train = 0
        loss_train_standard = 0 
        train_MSE = 0
        for i, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device).float(), y_train.to(device).float()
            y_pred = self.model.forward(X_train)
            
            if isauxiliary:
                y_var_pred = y_pred[:,:out_features]
                y_loss_pred = y_pred[:,-1]
                c_include = 1 
            else: 
                y_var_pred = y_pred
                y_loss_pred = 0
                c_include = 0

            loss_standard = self.criterion(y_var_pred, y_train)
            loss_pred = torch.abs(loss_standard-y_loss_pred)
            
            loss = loss_standard + c_include*loss_pred
            loss = torch.mean(loss, dim=0)
            
            loss_standard = torch.mean(loss_standard, dim=0)
            if self.L1reg: loss += self.l1_regularization()
            loss_train += loss.item()
            loss_train_standard += loss_standard.item()
            
            train_MSE += nn.MSELoss()(y_var_pred, y_train).item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                if self.epoch < epoch_th2: 
                    self.pickScheduler(
                        epoch_th1=epoch_th1, 
                        epoch_th2=epoch_th2
                    ).step()
                else: 
                    self.pickScheduler(
                        epoch_th1=epoch_th1, 
                        epoch_th2=epoch_th2
                    ).step(loss_train)
            self.learningrate = self.optimizer.param_groups[0]["lr"]

        loss_train_average = loss_train / len(train_loader)
        self.losses_train_list.append(loss_train_average)
        self.standard_losses_train_list.append(loss_train_standard/len(train_loader))
        self.MSE_train_list.append(train_MSE/len(train_loader))
        self.epoch += 1

    def eval_model(
        self, 
        train_loader, 
        val_loader, 
        test_loader,
        istrainloss=False, 
        istestloss=False, 
        isvalloss= True,
        isauxiliary=False, 
        means_out= None, 
        computeR2=False
    ):
        
        eval_func = self.MSELoss_c
        
        self.model.eval()
        with torch.no_grad():
            if istrainloss:
                
                loss_train_eval = 0
                train_MSE_eval = 0
                
                for i, (X_train, y_train) in enumerate(train_loader):
                    X_train, y_train = X_train.to(device), y_train.to(device)
                    y_pred = self.model.forward(X_train)
                    
                    if isauxiliary:
                        y_var_pred = y_pred[:,:out_features]
                    else: 
                        y_var_pred = y_pred
                        
                    loss_standard = eval_func(y_var_pred, y_train)
                    loss_standard = torch.mean(loss_standard, dim=0)
                    
                    loss_train_eval += loss_standard.item()
                    train_MSE_eval += nn.MSELoss()(y_var_pred, y_train).item()
                    
                self.MSE_train_eval_list.append(train_MSE_eval/len(train_loader))
                self.losses_train_eval_list.append(loss_train_eval/len(train_loader))
            
            if isvalloss:
                
                loss_val = 0 
                loss_val_standard = 0 
                mse = 0 
                variance = 0 
                total_samples = 0
                
                for i, (X_val, y_val) in enumerate(val_loader):
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    y_pred = self.model.forward(X_val)
                    
                    if isauxiliary:
                        y_var_pred = y_pred[:,:out_features]
                    else: 
                        y_var_pred = y_pred
                        
                    loss_standard = eval_func(y_var_pred, y_val)
                    loss_standard = torch.mean(loss_standard, dim=0)
                    
                    if computeR2: 
                        mse += torch.sum((y_pred-y_val)**2, dim=0)
                        variance += torch.sum(
                            (y_val-torch.tensor(means_out).to(device))**2,
                            dim=0
                        )
                        total_samples += y_val.size(0)
                        
                    loss_val += loss_standard.item()
                    
                self.losses_val_list.append(loss_val/len(val_loader))
                
                if computeR2:  
                    mse_mean = mse / total_samples
                    variance_mean = variance / total_samples
                    variance_mean[variance_mean == 0] = 1e-12
                    
                    r2 = 1 - (mse_mean / variance_mean).cpu().numpy()
                    self.r2_val_list.append(r2)

            if istestloss:
                loss_test = 0 
                for i, (X_test, y_test) in enumerate(test_loader):
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    y_pred = self.model.forward(X_test)
                    if isauxiliary:
                        y_pred = y_pred[:,:out_features]
                    else: 
                        y_pred = y_pred
                        
                    loss = eval_func(y_pred, y_test)
                    loss = torch.mean(loss, dim=0)
                    loss_test += loss.item()
                self.losses_test_list.append(loss_test/len(test_loader))


    def Train_eval_loop(
        self, 
        train_loader, 
        val_loader, 
        test_loader, 
        epochs=epochs,
        istrainloss=False, 
        istestloss=False, 
        isvalloss= True,
        isauxiliary=False, 
        epoch_th1=16, 
        epoch_th2=24
    ):
        
        self.model = self.model.to(device)
        
        for epoch in range(epochs):
            if self.totime: 
                start_t = time.time()
                
            self.train_model(
                train_loader, 
                isauxiliary=isauxiliary, 
                epoch_th1=epoch_th1, 
                epoch_th2=epoch_th2
            )
            
            if self.totime: 
                intermediate_t = time.time()
                
            self.eval_model(
                train_loader, 
                val_loader, 
                test_loader,
                istrainloss=istrainloss, 
                istestloss=istestloss, 
                isvalloss= isvalloss,
                isauxiliary=isauxiliary
            )
            
            if self.totime: final_t = time.time()
                
            print(f'''Epoch {epoch+ self.epoch_start}: ''')
            print(f'''\t Training Loss = {self.losses_train_list[-1]}''')
            
            if istrainloss: 
                print(f'''\t Training Loss (eval) = {self.losses_train_eval_list[-1]}''')
            if isvalloss: 
                print(f'''\t Validation Loss (eval) = {self.losses_val_list[-1]}''')
            if istestloss: 
                print(f'''\t Test Loss (eval) = {self.losses_test_list[-1]}''')
            if self.toprintlr: 
                print("Learning rate: ", self.learningrate)
            if self.totime: 
                print(
                    f"time evaluation loop : {final_t-intermediate_t}, time training loop: {intermediate_t-start_t} seconds"
                )

            ## in case the model should be saved every few steps
            # save_step = 5
            # if self.tosave and (epoch+ self.epoch_start)%save_step == 0: 
            #     torch.save(
            #         self.model, 
            #         "/home/eismaili/2_Thesis/Scripts/NN" +  f"/Saved_models/{self.model_title}_epoch{epoch+ self.epoch_start}.pt"
            #     )
            
            if self.tosave and (epoch + self.epoch_start) == epochs: 
                torch.save(
                    self.model, 
                    "/home/eismaili/Masters_Thesis" +  f"/Saved_models/{self.model_title}_epoch{epoch+ self.epoch_start}.pt"
                )



# Usage
# trainer = Trainer(model, criterion)
# trainer.Train_eval_loop(train_loader, val_loader, test_loader, epochs=10)
# print("Training losses:", trainer.losses_train_list)


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)  

    def forward(self, x):
        return self.linear(x) * self.mask


