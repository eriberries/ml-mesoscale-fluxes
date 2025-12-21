def Compute_R2_scores(y_pred, y_true, isnumpy=False):
    
    if isnumpy: 
        ss_res = np.sum((y_true-y_pred)**2, axis=0)
        y_mean = np.mean(y_true, axis=0)
        ss_tot = np.sum((y_true-y_mean)**2, axis=0)
        
        r2 = 1-(ss_res/ss_tot)
        r2_global = np.mean(r2)
        
    else:
        ss_res = torch.sum((y_true-y_pred)**2, dim=0)
        y_mean = torch.mean(y_true, dim=0)
        ss_tot = torch.sum((y_true-y_mean)**2, dim=0)
        
        r2 = 1-(ss_res/ss_tot)
        r2_global = (
            [torch.mean(r2[0*n_levs + 5 : (0+1)*n_levs])] 
            + [torch.mean(r2[i*n_levs : (i+1)*n_levs]) for i in range(1,4)]
        )

    return r2, ss_tot/len(y_true), ss_res/len(y_true), r2_global


def Compute_R2_scores_zonaltimemean(
    y_pred, 
    y_true, 
    isnumpy=False, 
    ismemory=ismemory, 
    extremes_th=extremes_th, 
    isremoveextr=False
):

    # split_ind = [None, None, slice(int(N_time*0.9), int(N_time))]
    
    if isremoveextr: 
        mask = return_mask(
            split_ind, 
            split = "test", 
            extremes_th=extremes_th, 
            X=X,
            discard10=discard10, 
            ismemory=ismemory
        )

        y_pred_hat = np.full((mask.shape[0], y_pred.shape[1]), np.nan, dtype=float)
        y_pred_hat[mask == True] = y_pred
        
        y_true_hat = np.full((mask.shape[0], y_true.shape[1]), np.nan, dtype=float)
        y_true_hat[mask == True] = y_true
        
    else:
        y_pred_hat = y_pred
        y_true_hat = y_true
    
    y_pred = y_pred_hat.T.reshape(
        y_pred_hat.shape[1], 
        int(N_time)-int(N_time*0.9), 
        N_lat, 
        N_lon
    )
    y_true = y_true_hat.T.reshape(
        y_true_hat.shape[1], 
        int(N_time)-int(N_time*0.9), 
        N_lat, 
        N_lon
    )
    
    ss_res = np.nansum((y_true-y_pred)**2, axis=(1,3)).T
    #res_size = ss_res.shape
    #print(res_size)
    y_mean = np.nanmean(y_true, axis=(1,3)).reshape(y_pred_hat.shape[1], 1, N_lat, 1)
    ss_tot = np.nansum((y_true-y_mean)**2, axis=(1,3)).T
    r2 = 1-(ss_res/ss_tot)
    
    return r2, ss_tot/y_true.shape[1]/N_lon, ss_res/y_true.shape[1]/N_lon


def Compute_R2_scores_map(
    y_pred, 
    y_true, 
    isnumpy=False, 
    ismemory=ismemory, 
    extremes_th=extremes_th, 
    isremoveextr=False
):

    #split_ind = [None, None, slice(int(N_time*0.9), int(N_time))]
    
    if isremoveextr==True: 
        mask = return_mask(
            split_ind, 
            split = "test", 
            extremes_th=extremes_th, 
            X=X,
            discard10=discard10, 
            ismemory=ismemory
        )

        y_pred_hat = np.full((mask.shape[0], y_pred.shape[1]), np.nan, dtype=float)
        y_pred_hat[mask == True] = y_pred
        
        y_true_hat = np.full((mask.shape[0], y_true.shape[1]), np.nan, dtype=float)
        y_true_hat[mask == True] = y_true
    else:
        y_pred_hat = y_pred
        y_true_hat = y_true
        
    N_time_hat = int(N_time)-int(N_time*0.9)
    
    y_pred = y_pred_hat.T.reshape(y_pred_hat.shape[1], N_time_hat, N_lat, N_lon)
    y_true = y_true_hat.T.reshape(y_true_hat.shape[1], N_time_hat, N_lat, N_lon)

    r2_list = []
    mse_list = []
    stdev_list = []

    for i in range(4):
        if i ==0: 
            ss_res = np.nansum(
                (y_true[5:n_levs]-y_pred[5:n_levs])**2, 
                axis=(0,1)
            ).T
        else:
            ss_res = np.nansum(
                (y_true[i*n_levs:(i+1)*n_levs]-y_pred[i*n_levs:(i+1)*n_levs])**2, 
                axis=(0,1)
            ).T
            
        # res_size = ss_res.shape
        # print(res_size)
        y_mean = np.nanmean(y_true, axis=(0,1)).reshape(1, 1, N_lat, N_lon)
        ss_tot = np.nansum((y_true[i*n_levs:(i+1)*n_levs]-y_mean)**2, axis=(0,1)).T
        
        mse_list.append(ss_res/N_lon/N_lat)
        stdev_list.append(ss_res/N_lon/N_lat)
        r2_list.append(1-(ss_res/ss_tot))
    
    return np.array(r2_list), np.array(stdev_list), np.array(mse_list)


def R2_levlatlon(true, pred, nonloc=True, numvars=4):
    
    var_list = []
    mse_list = []
    r2_list = []
    
    for i in range(numvars):
        
        if nonloc:
            nfeatlevs=22
        else:
            nfeatlevs=1
            
        tr = true[:,i*nfeatlevs:(i+1)*nfeatlevs]
        pr = pred[:,i*nfeatlevs:(i+1)*nfeatlevs]
        mse = np.mean((tr-pr)**2)
        var = tr.var()
        r2 = 1 - mse/var
        r2_list.append(r2)
    return np.array(var_list), np.array(mse_list), np.array(r2_list)


