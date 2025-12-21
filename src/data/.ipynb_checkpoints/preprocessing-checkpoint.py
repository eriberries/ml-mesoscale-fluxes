coarsedir = "../data/coarse"
normdir = "../data/normdata"

def make_splits_from_config(config):
    """
    Create slice objects for train/val/test splits
    """

    N_time = config["data"]["N_time"]
    s_train = config["data"]["size_split"]["train"]
    s_val   = config["data"]["size_split"]["val"]
    s_test  = config["data"]["size_split"]["test"]

    i_train_end = int(N_time * s_train)
    i_val_end   = int(N_time * (s_train + s_val))

    train_sel = slice(0, i_train_end)
    val_sel   = slice(i_train_end, i_val_end)
    test_sel  = slice(i_val_end, N_time)

    return {
        "train": train_sel,
        "val": val_sel,
        "test": test_sel,
    }


def include_memory(arr, N_time, Input=True, discard10=True):
    """
    Adds one-step temporal memory to flattened climate ML data.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (n_samples, n_features).
    N_time : int
        Number of time steps 
    Input : bool, default True
        If True, return stacked (current, previous) input fields.
        If False, return only the previous time step (for targets).
    discard10 : bool, default True
        If True, use 22 levels. Otherwise use all 32.

    Returns
    -------
    np.ndarray
        Memory-augmented array with shape:
        - Input=True: (2*n_vars, (N_time-1)*N_lat*N_lon)
        - Input=False: (n_vars, (N_time-1)*N_lat*N_lon)
        then transposed to (samples, features).
    """

    N_lat = 29
    N_lon = 47

    # reshape: (samples, features_flat) to (n_vars, time, space)
    # we know arr has shape (n_vars * N_time * N_lat*N_lon, ?), so reconstruct:
    n_vars = arr.shape[1] // (N_time * N_lat * N_lon)
    arr_reshaped = arr.T.reshape(n_vars, N_time, N_lat * N_lon)

    arr_future = arr_reshaped[:, 1:, :]   # current time t
    arr_past   = arr_reshaped[:, :-1, :]  # previous time t-1

    if Input:
        # stack along variable axis: shape = (2*n_vars, N_time-1, N_lat*N_lon)
        stacked = np.concatenate([arr_future, arr_past], axis=0)
        out = stacked.reshape(2 * n_vars, (N_time - 1) * N_lat * N_lon).T
    else:
        # for target variables: return only arr at t = 1,...,N_time-1
        out = arr_future.reshape(n_vars, (N_time - 1) * N_lat * N_lon).T

    return out



def load_split_intermediate_array(
    split_ind,
    split,
    Input,
    X,
    ismemory,
    discard10=True,
    whichMoisture="Q",
    r_c=(False, 0),
    bootstrap=False,
):
    """
    Collects the data into a numpy array without preprocessing.

    Parameters
    ----------
    split_ind : tuple(slice, slice, slice)
        (train_sel, val_sel, test_sel) time index slices.
    split : {"train", "val", "test"}
        Which split to load.
    Input : bool
        If True, load input variables. If False, load flux targets.
    X : int or str
        Identifier used in the filename (e.g. 5 for "X5").
    ismemory : bool
        If True, apply include_memory to the flattened data.
    discard10 : bool, default True
        If True, use lev=10:32 (22 levels). Otherwise use all 32 levels.
    whichMoisture : {"Q", "RH"}, default "Q"
        Moisture variable to use.
    r_c : tuple(bool, int), default (False, 0)
        (flag, cluster_id) to optionally remove a given cluster id.
    bootstrap : bool, default False
        If True, use ".bootstrap" suffix in filename.
    """

    if r_c[0] and bootstrap:
        raise ValueError(
            "You cannot set both 'r_c[0]' and 'bootstrap' to True, because no cluster labels are assigned to the additional bootstrapping test set"
        )

    # suffix for bootstrap files
    elif bootstrap:
        flag = ".bootstrap"
    else:
        flag = ""

    # grid dimensions (used later for cluster masking)
    N_lat = 29
    N_lon = 47
    n_levs = 22 if discard10 else 32

    # unpack split indices and pick the correct one
    train_sel, val_sel, test_sel = split_ind
    split_map = {
        "train": train_sel,
        "val": val_sel,
        "test": test_sel,
    }
    if split not in split_map:
        raise ValueError(f"Unknown split: {split!r}. Must be 'train', 'val', or 'test'.")
    time_indices = split_map[split]

    # moisture suffix in filename
    if whichMoisture == "RH":
        moisture_info = ".RH"
    elif whichMoisture == "Q":
        moisture_info = ""
    else:
        raise ValueError("whichMoisture must be 'Q' or 'RH'.")

    # open the dataset
    filename = (
        idir
        + f"X{X}.allyears.varsforML{moisture_info}.f09regridded{flag}.nc"
    )

    if Input:
        ds = (
            xr.open_dataset(filename)
                .isel(time=time_indices)
                .drop_vars([
                    "OMEGAQ_Flux", "OMEGAT_Flux", "OMEGAU_Flux", "OMEGAV_Flux"
                ])
        )
    else:
        ds = (
            xr.open_dataset(filename)
                .isel(time=time_indices)
                .drop_vars([
                    "T", "U", "V", whichMoisture, "OMEGA", "PS", "CAPE",
                    "dT_dx", "dT_dy", "dU_dx", "dU_dy", "dV_dx", "dV_dy",
                ])
        )   

    # vertical selection
    if discard10:
        ds = ds.isel(lev=slice(10, 32))

    # variable lists
    if Input:
        variables = [
            "U", "V", "OMEGA", "T", whichMoisture,
            "dU_dx", "dU_dy", "dV_dx", "dV_dy", "dT_dx", "dT_dy",
            "PS", "CAPE",
        ]
    else:
        variables = [
            "OMEGAQ_Flux",
            "OMEGAT_Flux",
            "OMEGAU_Flux",
            "OMEGAV_Flux",
        ]

    if Input:
        # profile variables (excl PS, CAPE)
        profile_vars = [v for v in variables if v not in ["PS", "CAPE"]]
        # surface variables
        surface_vars = [v for v in variables if v in ["PS", "CAPE"]]

        per_level_data = [
            ds[var].values[:, i, :, :]
            for var in profile_vars
            for i in range(n_levs)
        ]
        surface_data = [ds[var].values for var in surface_vars]

        data = np.array(per_level_data + surface_data)
    else:
        # outputs: expand over levels
        data = np.array(
            [
                ds[var].values[:, i, :, :]
                for var in variables
                for i in range(n_levs)
            ]
        )

    # flatten: (var, time, lat, lon) becomes (N_time*N_lat*N_lon, n_features)
    N_time = data.shape[1]
    data = data.reshape(
        data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]
    ).T

    # optional memory augmentation
    if ismemory:
        data = include_memory(data, N_time, Input=Input, discard10=discard10)

    # optional cluster removal
    if r_c[0]:
        s_flatten = slice(
            time_indices.start * N_lat * N_lon,
            time_indices.stop * N_lat * N_lon,
        )
        cluster = np.load(idir + "Cluster_Labels.npy")
        cluster_subset = cluster[s_flatten]
        original_shape = data.shape
        data = data[cluster_subset != r_c[1]]
        print(f"Cluster {r_c[1]} removed, data shape: {original_shape} --> {data.shape}")

    return data


def remove_extremes(
    split_ind,
    extremes_th,
    split,
    X,
    ismemory,
    discard10=True,
    whichMoisture="Q",
    r_c=(False, 0),
    bootstrap=False,
):
    """
    Remove samples whose standardized flux values in the output exceed threshold levels.

    Parameters
    ----------
    split_ind : tuple
        (train_sel, val_sel, test_sel) slices.
    extremes_th : list or float
        Thresholds (in standard deviations) beyond which samples are removed.
        If a list, must match number of variable groups (=4), and be in the same order 
        as the ouput fluxes.
        data ordering: [OMEGAQ_levs, OMEGAT_levs, OMEGAU_levs, OMEGAV_levs]
    split : {"train","val","test"}
        Which split to load.
    X : int or str
        Identifier used in filenames, e.g. X=5 -> "X5".
    ismemory : bool
        Whether memory augmentation is applied.
    discard10 : bool, default True
        Whether to discard the bottom 10 levels (use 22 levels instead of 32).
    whichMoisture : {"Q","RH"}, default "Q"
        Moisture variable used in input/output.
    r_c : tuple(bool, int), default (False,0)
        Cluster removal flag + cluster ID.
    bootstrap : bool, default False
        If True, load ".bootstrap" file.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask with True = keep, False = remove.
    """
    n_lev = 22 if discard10 else 32

    data_flux = load_split_intermediate_array(
        split_ind,
        split,
        Input=False,  
        X=X,
        ismemory=ismemory,
        discard10=discard10,
        whichMoisture=whichMoisture,
        r_c=r_c,
        bootstrap=bootstrap,
    )

    data_std = StandardScaler().fit_transform(data_flux)

    if isinstance(extremes_th, (int, float)):
        extremes = [extremes_th] * 4  # one threshold per flux variable
    else:
        if len(extremes_th) != 4:
            raise ValueError("extremes_th must be scalar or list of length 4.")
        extremes = extremes_th

    mask = np.ones(data_std.shape[0], dtype=bool)

    # Apply thresholds per flux variable block
    for i, th in enumerate(extremes):
        start = i * n_lev
        end = (i + 1) * n_lev
        mask &= np.all(np.abs(data_std[:, start:end]) <= th, axis=1)

    print("Extreme flux values exceeding threshold removed.")
    return mask



def normalize_data(
    split="train",
    Input=True,
    split_ind=split_ind,
    X=5,
    normalize="profile",
    discard10=True,
    isremoveextr=False,
    extremes_th=extremes_th,
    ismemory=False,
    whichMoisture=whichMoisture,
    r_c=(False, 0),
    bootstrap=False,
):
    """
    Load data for a given split and apply optional outlier removal
    and normalization.

    Parameters
    ----------
    split : {"train", "val", "test"}, default "train"
    Input : bool, default True
        If True, normalize inputs. If False, normalize outputs (fluxes).
    split_ind : tuple
        (train_sel, val_sel, test_sel) slices.
    X : int or str, default 5
        Identifier in the data filename (e.g. "X5").
    normalize : {"profile", "level", "momentum", "stdev_max", "stdev_min",
                 "stdev_mean", "stdev_level"}, default "profile"
    discard10 : bool, default True
        If True, use 22 levels; otherwise 32.
    isremoveextr : bool, default False
        If True, remove extreme samples before normalization.
    extremes_th : scalar or list, default extremes_th
        Threshold(s) used by Remove_Extremes.
    ismemory : bool, default False
        If True, IncludeMemory was applied to data.
    whichMoisture : {"Q", "RH"}, default whichMoisture
    r_c : tuple(bool, int), default (False, 0)
    bootstrap : bool, default False

    Returns
    -------
    data : np.ndarray
        Normalized data array.
    """

    eps = 1e-12
    n_levs = 22 if discard10 else 32

    # ------------------------------------------------------------------
    # Load data 
    # ------------------------------------------------------------------
    base_data = load_split_intermediate_array(load_split_intermediate_array
        split_ind=split_ind,
        split=split,
        Input=Input,
        X=X,
        ismemory=ismemory,
        discard10=discard10,
        whichMoisture=whichMoisture,
        r_c=r_c,
        bootstrap=bootstrap,
    )

    if isremoveextr:
        mask = remove_extremes(
            split_ind=split_ind,
            extremes_th=extremes_th,
            split=split,
            X=X,
            ismemory=ismemory,
            discard10=discard10,
            whichMoisture=whichMoisture,
            r_c=r_c,
            bootstrap=bootstrap,
        )
        data = base_data[mask]
    else:
        data = base_data

    # ------------------------------------------------------------------
    # OUTPUT (flux) normalizations
    # ------------------------------------------------------------------
    if not Input:

        # adjusting to momentum scale
        if normalize == "momentum":
            c_p = 1.8 * 0.05 / 0.025
            L_v = 3 * 0.05 / 0.5 * 10**4  
            const_list = [10 * L_v, 10 * c_p, 10, 10]  

            for i in range(4):
                s = i * n_levs
                e = (i + 1) * n_levs
                data[:, s:e] = const_list[i] * data[:, s:e]

        # stdev-based scaling: max, min, mean
        if normalize == "stdev_max":
            fname = normdir + "output_train_profile_stdev_max.npy"
            if split == "train":
                stdev_arr = np.array(
                    [
                        np.max(np.std(data[:, i * n_levs:(i + 1) * n_levs], axis=0))
                        for i in range(4)   
                    ]
                ) + eps
                np.save(fname, stdev_arr)
            else:
                stdev_arr = np.load(fname)

            for i in range(4):
                s = i * n_levs
                e = (i + 1) * n_levs
                data[:, s:e] = data[:, s:e] / stdev_arr[i]

        if normalize == "stdev_min":
            fname = normdir + "output_train_profile_stdev_min.npy"
            if split == "train":
                stdev_arr = np.array(
                    [
                        np.min(np.std(data[:, i * n_levs:(i + 1) * n_levs], axis=0))
                        for i in range(4)
                    ]
                ) + eps
                np.save(fname, stdev_arr)
            else:
                stdev_arr = np.load(fname)

            for i in range(4):
                s = i * n_levs
                e = (i + 1) * n_levs
                data[:, s:e] = data[:, s:e] / stdev_arr[i]

        if normalize == "stdev_mean":
            fname = normdir + "output_train_profile_stdev_mean.npy"
            if split == "train":
                stdev_arr = np.array(
                    [
                        np.mean(np.std(data[:, i * n_levs:(i + 1) * n_levs], axis=0))
                        for i in range(4)
                    ]
                ) + eps
                np.save(fname, stdev_arr)
            else:
                stdev_arr = np.load(fname)

            for i in range(4):
                s = i * n_levs
                e = (i + 1) * n_levs
                data[:, s:e] = data[:, s:e] / stdev_arr[i]

        if normalize == "stdev_level":
            fname = normdir + "output_train_profile_std_stdev_level.npy"
            if split == "train":
                stdev_arr = np.std(data, axis=0) + eps
                np.save(fname, stdev_arr)
            else:
                stdev_arr = np.load(fname)

            for i in range(4 * n_levs):
                data[:, i] = data[:, i] / stdev_arr[i]

    # ------------------------------------------------------------------
    # INPUT normalizations
    # ------------------------------------------------------------------
    if Input:

        # profile normalization (per variable, then add PS, CAPE)
        if normalize == "profile":
            if split == "train":
                means = []
                stds = []

                # 11 profile variables, each with n_levs levels
                for i in range(11):
                    s = i * n_levs
                    e = (i + 1) * n_levs
                    d_mean = np.mean(data[:, s:e])
                    d_std = np.std(data[:, s:e]) + eps
                    data[:, s:e] = (data[:, s:e] - d_mean) / d_std
                    means.append(d_mean)
                    stds.append(d_std)

                # PS and CAPE (single level)
                idx_ps = 11 * n_levs
                d_mean = np.mean(data[:, idx_ps])
                d_std = np.std(data[:, idx_ps]) + eps
                data[:, idx_ps] = (data[:, idx_ps] - d_mean) / d_std
                means.append(d_mean)
                stds.append(d_std)

                idx_cape = 11 * n_levs + 1
                d_mean = np.mean(data[:, idx_cape])
                d_std = np.std(data[:, idx_cape]) + eps
                data[:, idx_cape] = (data[:, idx_cape] - d_mean) / d_std
                means.append(d_mean)
                stds.append(d_std)

                # memory part
                if ismemory:
                    start = 11 * n_levs + 2
                    # 11 profile variables for memory
                    for i in range(11):
                        s = start + i * n_levs
                        e = start + (i + 1) * n_levs
                        d_mean = np.mean(data[:, s:e])
                        d_std = np.std(data[:, s:e]) + eps
                        data[:, s:e] = (data[:, s:e] - d_mean) / d_std
                        means.append(d_mean)
                        stds.append(d_std)

                    # PS (memory)
                    idx_ps_m = start + 11 * n_levs
                    d_mean = np.mean(data[:, idx_ps_m])
                    d_std = np.std(data[:, idx_ps_m]) + eps
                    data[:, idx_ps_m] = (data[:, idx_ps_m] - d_mean) / d_std
                    means.append(d_mean)
                    stds.append(d_std)

                    # CAPE (memory)
                    idx_cape_m = start + 11 * n_levs + 1
                    d_mean = np.mean(data[:, idx_cape_m])
                    d_std = np.std(data[:, idx_cape_m]) + eps
                    data[:, idx_cape_m] = (data[:, idx_cape_m] - d_mean) / d_std
                    means.append(d_mean)
                    stds.append(d_std)

                np.save(normdir + "input_train_profile_means.npy", np.array(means))
                np.save(normdir + "input_train_profile_stds.npy", np.array(stds))

            else:
                means = np.load(normdir + "input_train_profile_means.npy")
                stds = np.load(normdir + "input_train_profile_stds.npy")

                # 11 profile variables
                for i in range(11):
                    s = i * n_levs
                    e = (i + 1) * n_levs
                    data[:, s:e] = (data[:, s:e] - means[i]) / stds[i]

                # PS and CAPE (use last two means/stds)
                data[:, 11 * n_levs] = (
                    data[:, 11 * n_levs] - means[-2]
                ) / stds[-2]
                data[:, 11 * n_levs + 1] = (
                    data[:, 11 * n_levs + 1] - means[-1]
                ) / stds[-1]

        # level-wise normalization: each feature separately
        elif normalize == "level":
            if split == "train":
                means = []
                stds = []
                for i in range(data.shape[1]):
                    d_mean = np.mean(data[:, i])
                    d_std = np.std(data[:, i]) + eps
                    data[:, i] = (data[:, i] - d_mean) / d_std
                    means.append(d_mean)
                    stds.append(d_std)

                np.save(normdir + "input_train_level_means.npy", np.array(means))
                np.save(normdir + "input_train_level_stds.npy", np.array(stds))
            else:
                means = np.load(normdir + "input_train_level_means.npy")
                stds = np.load(normdir + "input_train_level_stds.npy")
                for i in range(data.shape[1]):
                    data[:, i] = (data[:, i] - means[i]) / stds[i]

    return data


def Save_to_numpy(
    split="train",
    Input=True,
    split_ind=None,
    X=5,
    normalize="profile",
    discard10=True,
    isremoveextr=True,
    extremes_th=None,
    ismemory=False,
    crossval=False,
    isfinal=False,
    whichMoisture="Q",
    r_c=(False, 0),
    bootstrap=False,
):
    """
    Normalize data for a given split and save it as a .npy file.

    Parameters
    ----------
    split : {"train", "val", "test"}, default "train"
    Input : bool, default True
        True for inputs, False for outputs.
    split_ind : tuple or None
        (train_sel, val_sel, test_sel). Must be provided.
    X : int or str, default 5
        Identifier in filenames, e.g. 'X5'.
    normalize : str, default "profile"
        Normalization scheme (see normalize_data).
    discard10 : bool, default True
        Whether to discard lower levels (use 22 instead of 32).
    isremoveextr : bool, default True
        Whether to remove extremes via Remove_Extremes.
    extremes_th : scalar or list, default None
        Threshold(s) passed to Remove_Extremes.
    ismemory : bool, default False
        Whether memory augmentation is used.
    crossval : bool or int, default False
        False/True or fold index. If not bool, treated as k-fold index.
    isfinal : bool, default False
        If True, save only final training set.
    whichMoisture : {"Q", "RH"}, default "Q"
    r_c : tuple(bool, int), default (False, 0)
        Cluster removal flag + cluster ID.
    bootstrap : bool, default False
    """
    if split_ind is None:
        raise ValueError("split_ind must be provided to Save_to_numpy.")
    if extremes_th is None:
        raise ValueError("extremes_th must be provided to Save_to_numpy.")

    data = normalize_data(
        split=split,
        Input=Input,
        split_ind=split_ind,
        X=X,
        normalize=normalize,
        discard10=discard10,
        isremoveextr=isremoveextr,
        extremes_th=extremes_th,
        ismemory=ismemory,
        whichMoisture=whichMoisture,
        r_c=r_c,
        bootstrap=bootstrap,
    )

    # base title
    if Input:
        title = f"NN_data_{split}_input_normalized_{normalize}"
    else:
        title = f"NN_data_{split}_output_normalized_{normalize}"

    discard10_str = "discard10" if discard10 else "keep10"
    extremes_str = "withExtremes" if not isremoveextr else "noExtremes"
    if isremoveextr:
        extremes_str += f"{extremes_th}"
    ismemory_str = "withMemory" if ismemory else "noMemory"
    bootstrap_str = "_bootstrap" if bootstrap else ""
    exclc_str = f"_nocluster{r_c[1]}" if r_c[0] else ""

    suffix = (
        f"_{whichMoisture}"
        f"_{discard10_str}"
        f"_{extremes_str}"
        f"_{ismemory_str}"
        f"_X{X}"
        f"{exclc_str}"
        f"{bootstrap_str}"
    )

    title += suffix

    # cross-validation fold index
    if not isinstance(crossval, bool):
        if X == 15 or isfinal:
            raise Exception(
                "Final (or the large X=15) dataset is not for crossvalidation!"
            )
        title += f"_k{crossval}"

    # final training set marker
    if isfinal and split == "train":
        title += "_final"

    filename = title + ".npy"
    np.save(idir + filename, data)
    print(f'Saved file "{filename}" of size {data.nbytes / 1e9:.3f} GB')


def return_mask(
    split_ind,
    split="test",
    extremes_th=None,
    X=5,
    discard10=True,
    ismemory=False,
    whichMoisture="Q",
    r_c=(False, 0),
    bootstrap=False,
):
    """
    Convenience wrapper to return the boolean mask from Remove_Extremes
    for a given split.
    """
    if extremes_th is None:
        raise ValueError("extremes_th must be provided to return_mask.")

    mask = remove_extremes(
        split_ind=split_ind,
        extremes_th=extremes_th,
        split=split,
        X=X,
        ismemory=ismemory,
        discard10=discard10,
        whichMoisture=whichMoisture,
        r_c=r_c,
        bootstrap=bootstrap,
    )
    return mask


def Save_tvt(
    split_ind=None,
    X=5,
    normalize_in_list=None,
    normalize_out_list=None,
    discard10=True,
    isremoveextr=True,
    extremes_th=None,
    ismemory=False,
    crossval=False,
    isfinal=False,
    onlyinput=False,
    onlyoutput=False,
    whichMoisture="Q",
    r_c=(False, 0),
):
    """
    Save train/val/test (or train/val for CV) numpy files for inputs/outputs
    and multiple normalization schemes.

    Parameters
    ----------
    split_ind : tuple or None
        (train_sel, val_sel, test_sel). Must be provided.
    X : int or str, default 5
    normalize_in_list : list of str or None
        Normalizations for inputs. Default ["level"].
    normalize_out_list : list of str or None
        Normalizations for outputs. Default ["momentum"].
    discard10, isremoveextr, extremes_th, ismemory, crossval, isfinal,
    onlyinput, onlyoutput, whichMoisture, r_c :
        See Save_to_numpy / normalize_data.
    """
    if split_ind is None:
        raise ValueError("split_ind must be provided to Save_tvt.")
    if extremes_th is None:
        raise ValueError("extremes_th must be provided to Save_tvt.")

    if normalize_in_list is None:
        normalize_in_list = ["level"]  
    if normalize_out_list is None:
        normalize_out_list = ["momentum"]

    if onlyinput and onlyoutput:
        raise Exception(
            "You cannot set onlyinput and onlyoutput both to True. "
            "Set only one to True or both to False (default)."
        )

    # final model: only save train split for both Input=True/False
    if isfinal:
        for Input in [True, False]:
            normalize_list = normalize_in_list if Input else normalize_out_list
            for normalize in normalize_list:
                Save_to_numpy(
                    split="train",
                    Input=Input,
                    split_ind=split_ind,
                    X=X,
                    normalize=normalize,
                    discard10=discard10,
                    isremoveextr=isremoveextr,
                    extremes_th=extremes_th,
                    ismemory=ismemory,
                    crossval=crossval,
                    isfinal=isfinal,
                    whichMoisture=whichMoisture,
                    r_c=r_c,
                )
        return

    # which of inputs/outputs to process?
    if not onlyinput and not onlyoutput:
        input_flags = [True, False]
    elif onlyinput:
        input_flags = [True]
    else:  # onlyoutput
        input_flags = [False]

    # splits: for normal case, do train/val/test; for CV or crossval=True, do train/val only
    if not crossval and isinstance(crossval, bool):
        split_list = ["train", "val", "test"]
    else:
        split_list = ["train", "val"]

    for Input in input_flags:
        normalize_list = normalize_in_list if Input else normalize_out_list
        for normalize in normalize_list:
            for split in split_list:
                Save_to_numpy(
                    split=split,
                    Input=Input,
                    split_ind=split_ind,
                    X=X,
                    normalize=normalize,
                    discard10=discard10,
                    isremoveextr=isremoveextr,
                    extremes_th=extremes_th,
                    ismemory=ismemory,
                    crossval=crossval,
                    isfinal=isfinal,
                    whichMoisture=whichMoisture,
                    r_c=r_c,
                )

