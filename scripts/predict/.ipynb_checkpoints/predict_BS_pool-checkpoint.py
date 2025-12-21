import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import threadpoolctl
threadpoolctl.threadpool_limits(limits=16)

#from List_of_directories import datadir, modeldir
#from config import *
from Sets_dict import experiment_info, cat_keys

M = 5_205_297 # size of bootstrap pool
n = 997_716 # size of the test set
M = M+n # size of bootstrap pool incl test set

for cat in cat_keys[:-1]:

    modelname = experiment_info[cat]["modelname"]
    print("MODELNAME: ", modelname)
    model = torch.load(modeldir + f"{modelname}.pt", weights_only=False, map_location=device).to(device)
    
    var_incl = experiment_info[cat]["var_incl"]
    
    
    class MyDataset(Dataset):
        def __init__(self, X, y, transform=None):
            self.X = X
            self.y = y
            self.n_samples = X.shape[0]
            self.transform = transform
    
        def __getitem__(self, index):
            sample = self.X[index], self.y[index]
            #print("Shape of inputs:", sample[0].shape)
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample 
    
        def __len__(self):
            return self.n_samples
    
    
    class ToTensor:
        def __call__(self, sample):
            inputs, target = sample
            #print(type(inputs), inputs.shape)
            return torch.from_numpy(inputs), torch.from_numpy(target)
            
    def Get_datasets(I = 0,
                     fn = [f"NN_data_val_input_normalized_level_Q_discard10_withExtremes_noMemory_X5_bootstrap3.npy",
                           f"NN_data_val_output_normalized_momentum_Q_discard10_withExtremes_noMemory_X5_bootstrap3.npy"]):
        dataset_list = []
    
    
        variables_in = ["U", "V", "OMEGA", "T", "Q", "dU_dx", "dU_dy","dV_dx", "dV_dy", "dT_dx", "dT_dy", "PS", "CAPE"]
        nlevs = 22 if discard10 else 32
        print("var_incl: ", var_incl)
        print("variables_in: ", variables_in)
        var_indices = np.concatenate([
            np.arange(
                11*nlevs    if idx == 11 else
                11*nlevs+1  if idx == 12 else
                idx * nlevs,
                11*nlevs+1  if idx == 11 else
                11*nlevs+2  if idx == 12 else
                (idx+1) * nlevs
            )
            for idx in map(variables_in.index, var_incl)
        ])
        var_indices = np.sort(var_indices)
        
        if I<=5:
            X = np.take(np.load(datadir + fn[0]).astype(np.float32)[(I)*n:(I+1)*n], var_indices, axis=1)
            y = np.load(datadir + fn[1]).astype(np.float32)[(I)*n:(I+1)*n]
            print("first")
            print("X = ", datadir + fn[0])
            print("y = ", datadir + fn[0])
    
        else:
            X = np.take(np.load(datadir + fn[0]).astype(np.float32)[(I)*n:], var_indices, axis=1)
            y = np.load(datadir + fn[1]).astype(np.float32)[(I)*n:]
            print("second")
            print("X = ", datadir + fn[0])
            print("y = ", datadir + fn[0])
        
        dataset = MyDataset(X, y, transform=ToTensor())
        dataset_list.append(dataset)
    
        return dataset_list
    
    
    
    def Get_Prediction(model, loader):
        X = []
        predicted = []
        true = []
        model.eval()
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(loader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model.forward(X_val)
                X.append(X_val)
                predicted.append(y_pred)
                true.append(y_val)
        return torch.cat(X, dim=0).numpy(), torch.cat(true, dim=0).numpy(), torch.cat(predicted, dim=0).numpy()
    
    dataset = Get_datasets(I=0)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true0, predicted0 = Get_Prediction(model, loader)
    print("0")
    dataset = Get_datasets(I=1)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true1, predicted1 = Get_Prediction(model, loader)
    print("1")
    dataset = Get_datasets(I=2)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true2, predicted2 = Get_Prediction(model, loader)
    print("2")
    dataset = Get_datasets(I=3)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true3, predicted3 = Get_Prediction(model, loader)
    print("3")
    dataset = Get_datasets(I=4)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true4, predicted4 = Get_Prediction(model, loader)
    print("4")
    dataset = Get_datasets(I=5)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true5, predicted5 = Get_Prediction(model, loader)
    print("5")
    dataset = Get_datasets(I=6)[0]
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    _, true6, predicted6 = Get_Prediction(model, loader)
    print("6")
    
    np.save(datadir+ f"predicted_BSpool_{cat}.npy", np.concatenate([predicted0, predicted1, predicted2, predicted3, predicted4, predicted5, predicted6], axis=0))
    
    # np.save(datadir+ f"predicted_BSpool_{cat}_blah.npy", predicted0)
    print(datadir+ f"predicted_BSpool_{cat}.npy")
