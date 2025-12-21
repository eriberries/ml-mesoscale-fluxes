import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import threadpoolctl
threadpoolctl.threadpool_limits(limits=16)

#from List_of_directories import datadir, modeldir
#from config import *
from Sets_dict_local import experiment_info, cat_keys

reps = 11
M = 136_466_286 # size of bootstrap pool
n = int(997_716*22/reps) # 


class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample 

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)


def Get_datasets(inputfile, outputfile, I = 0):
    dataset_list = []


    variables_in = [
        "U", "V", "OMEGA", "T", "Q", 
        "dU_dx", "dU_dy","dV_dx", "dV_dy", "dT_dx", "dT_dy", 
        "PS", "CAPE"
    ]
    nlevs = 3 if localization_type == "nnlev" else 1
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

    
    if I<=67:
        print(n)
        X = np.take(inputfile[(I)*n:(I+1)*n], var_indices, axis=1)
        y = outputfile[(I)*n:(I+1)*n]
    else:
        X = np.take(inputfile[(I)*n:], var_indices, axis=1)
        y = outputfile[(I)*n:]
    
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


for cat in cat_keys[1:]:
    modelname = experiment_info[cat]["modelname"]
    model = torch.load(
        modeldir + f"{modelname}.pt", 
        weights_only=False, 
        map_location=device
    ).to(device)
    var_incl = experiment_info[cat]["var_incl"]
    
    localization_type = "nnlev" if "nn_to_lev" in modelname else "1lev"
    fn = [
        f"NN_data_val_input_normalized_level_Q_discard10_withExtremes_noMemory_X5_bootstrap3_{localization_type}.npy",
        f"NN_data_val_output_normalized_momentum_Q_discard10_withExtremes_noMemory_X5_bootstrap3_1lev.npy"
    ]
    
    inputfile = np.load(datadir + fn[0]).astype(np.float32)
    outputfile = np.load(datadir + fn[1]).astype(np.float32)
    
    predicted_list = []
    for i in range(69):
        dataset = Get_datasets(inputfile, outputfile, I=i)[0]
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
        _, _, predicted = Get_Prediction(model, loader)
        predicted_list.append(predicted)
        print(f"size {predicted.nbytes/1e9} GB")
        print(i)
    
    np.save(datadir+ f"predicted_BSpool_{cat}_{localization_type}.npy", np.concatenate(predicted_list, axis=0))
    print(datadir+ f"predicted_BSpool_{cat}_{localization_type}.npy")
