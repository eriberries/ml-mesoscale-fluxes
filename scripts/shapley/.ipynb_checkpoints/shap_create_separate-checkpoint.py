import numpy as np
import torch
from torch.utils.data import DataLoader
import shap


# from List_of_directories import datadir, shapdir, modeldir
# from prepare_dataset import Get_datasets
# from config import normalize, normalize_output, batchsize, num_workers, device, discard10, seed_worker, g 
# from DS_info import levs


data_name_list = ["NN_data_test_input_normalized_level_Q_discard10_withExtremes_noMemory_X5.npy",
                  "NN_data_test_output_normalized_momentum_Q_discard10_withExtremes_noMemory_X5.npy"]
    
def Compute_subsample_shap(model, N_samples = 1000):
    
    N_set =  len(np.load(datadir + data_name_list[1]))
    
    ind_subsamples = np.random.choice([i for i in range(N_set)], int(1.5*N_samples))
    X = np.load(datadir + data_name_list[0])[ind_subsamples]
    y = np.load(datadir + data_name_list[1])[ind_subsamples]
    
    background = X[:N_samples]
    test = X[N_samples:]
    e = shap.DeepExplainer(model, torch.tensor(background).to(dtype=torch.float32))
    shap_values = e.shap_values(torch.tensor(test).to(dtype=torch.float32))
    
    return shap_values, X, y


reps = 10

for modelname in ["mainmodel"]: # ["Model_a", "Model_b", "Model_c", "Model_d", "Model_e"]:

    model = torch.load(modeldir + f"{modelname}_epoch16.pt", weights_only=False, map_location=device) 
    
    for i in range(reps):
        shap_values, X, y = Compute_subsample_shap(model)
        np.save(shapdir + f"shap_numpy_arrays/{modelname}_X_sub{i}.npy", X)
        np.save(shapdir + f"shap_numpy_arrays/{modelname}_y_sub{i}.npy", y)
        np.save(shapdir + f"shap_numpy_arrays/{modelname}_shapvalues_sub{i}.npy", shap_values)

