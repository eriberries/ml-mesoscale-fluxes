import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# from config import ismemory, discard10, X, n_levs, extremes_th, whichMoisture #config in parser and parser here --- CHANGE THIS
# 
# X_days = X #the input sets will be also called X therefore this variable is redefined here

# idir = "/net/krypton/climdyn/eismaili/Thesis/data/"

size_data = {}

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


def get_filename(
    Input, 
    split, 
    normalize, 
    discard10, 
    isremoveextr, 
    ismemory, 
    crossval, 
    X_days, 
    isfinal, 
    conformal, 
    r_c
):
    
    if Input: 
        title = f"NN_data_{split}_input_normalized_{normalize}"  
    else: 
        title = f"NN_data_{split}_output_normalized_{normalize}"

    discard10_str = "discard10" if discard10 else "keep10"
    extremes_str = "withExtremes" if not isremoveextr else "noExtremes"
    extremes_str += f"{extremes_th}" if extremes_str == "noExtremes" else ""
    ismemory_str = "withMemory" if ismemory else "noMemory"
    conformal_str = "_conformal" if conformal else ""
    exclc_str = f"_nocluster{r_c[1]}" if r_c[0] else ""
    
    string_for_filename = f"_{whichMoisture}_{discard10_str}_{extremes_str}_{ismemory_str}_X{X_days}{exclc_str}{conformal_str}"
    title += string_for_filename
    
    if not isinstance(crossval, bool): 
        title = title + f"_k{crossval}"
        if X_days == 15 or isfinal: 
            raise Exception('Final (or the large X=15) dataset is not for crossvalidation!')
    
    if isfinal and split == "train": 
        title = title + "_final"
    
    return title


def Get_datasets(
    variables_toremove = None, 
    indices_toremove = None,
    normalize_input="level", 
    normalize_output="momentum", 
    print_shape = True, 
    which_fold=(1,1,1),
    k=False, 
    isfinal=False, 
    discard10=discard10, 
    ismemory=ismemory,
    which_out_var = "all", 
    isremoveextr=False, 
    whichMoisture = "Q",
    inlocalization = "", 
    outlocalization = "", 
    additional_features="",
    div = False, 
    rot = False,
    size_data=None, 
    means_out=None, # additional information about the data that might be needed externally
    conformal=False, 
    r_c = (False, 0)
):

    n_levs = 22 if discard10 else 32
    
    if variables_toremove is not None:
        variables_in = [
            "U", "V", "OMEGA", "T", "Q", 
            "dU_dx", "dU_dy","dV_dx", "dV_dy", "dT_dx", "dT_dy", 
            "PS", "CAPE"
        ]
        
        slice_list = []
        for var_r in variables_toremove:
            var_index = variables_in.index(var_r)
            if var_index == 11: 
                var_slice = (11*n_levs, 11*n_levs+1)
            elif var_index == 12: 
                var_slice = (11*n_levs+1, 11*n_levs+2)
            else: 
                var_slice = (var_index*n_levs, (var_index+1)*n_levs) 
            slice_list.append(var_slice)
            
        var_slice = np.r_[tuple(np.s_[start:stop] for start, stop in slice_list)]
        print(var_slice)

    output_selection_slices = {
        "all": slice(None, None), 
        "OMEGAQ": slice(None, n_levs), 
        "OMEGAT": slice(n_levs,2*n_levs),
        "OMEGAU": slice(2*n_levs, 3*n_levs), 
        "OMEGAV": slice(3*n_levs, 4*n_levs) 
    }

    dataset_list = [] # the list containing the data [t,v,t] outputs
    
    ### idir = 

    fold_name = ["train", "val", "test"]
    
    for f in range(3):
        
        if which_fold[f]:
            split = fold_name[f]

            # Load input X
            normalize = normalize_input
            title_X = get_filename(
                True, split, normalize, discard10, isremoveextr, 
                ismemory, k, X_days, isfinal, conformal, r_c
            )
            print(f'Filename: "{title_X}{inlocalization}{additional_features}.npy"')
            
            if variables_toremove is not None:
                
                X = np.delete(
                    np.load(
                        idir + f"{title_X}{inlocalization}{additional_features}.npy"
                    ).astype(np.float32),
                    var_slice, 
                    axis=1
                )
                
            elif indices_toremove is not None:
                X = np.delete(
                    np.load(
                        idir + f"{title_X}{inlocalization}{additional_features}.npy"
                    ).astype(np.float32),
                    indices_toremove, 
                    axis=1
                )
            else: 
                X = np.load(
                    idir + f"{title_X}{inlocalization}{additional_features}.npy"
                ).astype(np.float32)

            # Load output y
            normalize = normalize_output
            title_y = get_filename(
                False, split, normalize, discard10, isremoveextr, 
                ismemory, k, X_days, isfinal, conformal, r_c
            )
            print(f'Filename: "{title_y}{outlocalization}.npy"')
            
            sel_slice = output_selection_slices[which_out_var]
            y = np.load(idir + f"{title_y}{outlocalization}.npy")[:, sel_slice]

            # Create torch Dataset
            dataset = MyDataset(X, y, transform=ToTensor())
            dataset_list.append(dataset)

            # print and add information:
            if f == 2 and means_out is not None: 
                means_out["mean"] = np.mean(y, axis=0)

            if print_shape:
                print(f"Data type of the {split} set (input, ouput): ", X.dtype, y.dtype)
                print(f"Shape of the {split} set (input, ouput): ", X.shape, y.shape)
                print(f"Size in GB:\n input {split} data: {X.nbytes/1e9}\n output validation data: {y.nbytes/1e9}")
        
            if size_data is not None: 
                size_data[f"{split}_X.shape"] = X.shape
                size_data[f"{split}_y.shape"] = y.shape

    return dataset_list

