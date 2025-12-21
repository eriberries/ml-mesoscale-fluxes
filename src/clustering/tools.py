def Elbow_plot(mydata, max_k, method=KMeans):
    inertias = []

    for k in range(1, max_k):
        print("start:",k)
        clusters = method(n_clusters=k, n_init=10)
        clusters.fit(mydata)
        
        inertias.append(clusters.inertia_)
        print("finish:",k)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(np.arange(1, max_k), inertias, 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

def SilhoutteScore_plot(mydata, k_range=(2,10), method=KMeans):
    scores = []
    min_k = k_range[0]
    max_k = k_range[1]
    for k in range(min_k, max_k):
        print("start:",k)
        clusters = method(n_clusters=k, n_init=10)
        clusters.fit(mydata)
        labels = clusters.labels_
        
        score = silhouette_score(mydata, labels)
        scores.append(score)
        print("finish:",k)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(np.arange(2, max_k), scores, 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhoutte Score")
    plt.grid(True)
    plt.show()


def Random_subsample(data, Random_way):
    # Select just randomly points from the entire sample
    if Random_way ==0 :
        ind_rand = np.random.choice(np.arange(data.shape[0]), N_small) # 1 ... Nlat*Nlon
        data_small = data[ind_rand,:]
        print(f'chosen total number of samples: {N_small}')
        print("\n shape: ", data_small.shape)
    
    
    # Select for each time step a constant number of random grid points
    if Random_way ==1 :
        data2 = np.concatenate(np.transpose(data_og, (2,3,1,0)))
        N_tstep = N_small//data2.shape[1]+1
        print(f'chosen total number of samples: {N_small}, total number of samples per timestep: {N_lat*N_lon}, chosen number of samples per timestep: {N_tstep}')
        data_small = np.array([data2[np.random.choice([i for i in range(N_lat*N_lon)], N_tstep) ,t,:] for t in range(N_time)])
        print("\n shape before reshape: ", data_small.shape, "\n number of gridpoints per time step: ", data_small.shape[1])
        data_small = np.concatenate(data_small)
        print("\n shape after reshape: ",data_small.shape)
        
    return data_small


def create_data_frame():
    ds = xr.open_dataset(datadir + "X5.allyears.varsforML.RH.f09regridded.nc")
    
    times, levs, lats, lons = ds.time.values, ds.lev.values, ds.lat.values, ds.lon.values
    N_time, N_lev, N_lat, N_lon = len(times), len(levs), len(lats), len(lons)

    Months = [i.month for i in ds.time.values]
    Season_list = ["DJF", "MAM", "JJA", "SON"]
    
    def Season_decide(month): 
        if month in [12, 1, 2]:
            season = Season_list[0]
        elif month in [3,4,5]:
            season = Season_list[1]
        elif month in [6,7,8]:
            season = Season_list[2]
        elif month in [9,10,11]:
            season = Season_list[3]
        return season
        
    Seasons = [Season_decide(i) for i in Months]
    
    Coordinates = [(lat, lon) for lat in lats for lon in lons]
    Coord_Season = [(coord, season) for season in Seasons for coord in Coordinates]
    
    Coordinates_ifconcantenated = [(lat, lon) for season in Seasons for lat in lats for lon in lons]
    Season_ifconcantenated = [season for season in Seasons for lat in lats for lon in lons]
    #times_ifconcatenated = [str(time) for time in times for lat in lats for lon in lons]
    
    data = np.array([
        ds.U.sel(lev=500, method='nearest').values - ds.U.sel(lev=850, method='nearest').values,
        ds.dU_dx.sel(lev=1000, method='nearest').values + ds.dV_dy.sel(lev=1000, method='nearest').values, # *a,
        ds.V.sel(lev=500, method='nearest').values - ds.V.sel(lev=850, method='nearest').values, 
        ds.T.sel(lev=1000, method='nearest').values - ds.T.sel(lev=850, method='nearest').values , 
        ds[whichMoisture].sel(lev=1000, method='nearest').values, 
        ds.PS.values,
        ds.CAPE.values,
        ds.OMEGAU_Flux.sel(lev=500, method='nearest').values,
        ds.OMEGAV_Flux.sel(lev=500, method='nearest').values,
        ds.OMEGAT_Flux.sel(lev=500, method='nearest').values,
        ds.OMEGAQ_Flux.sel(lev=500, method='nearest').values,
        ds.dV_dx.sel(lev=500, method='nearest').values - ds.dU_dy.sel(lev=500, method='nearest').values,
        ds.V.sel(lev=1000, method='nearest').values,
        (da_theta_e.sel(lev=500, method='nearest') - da_theta_e.sel(lev=850, method='nearest')).values,
        CSI_diag.values.swapaxes(0,1),
        Ri.values
    ])
    data = data.reshape(-1, data.shape[1]*data.shape[2]*data.shape[3]).T
    data_scaled = StandardScaler(with_mean=False).fit_transform(data)
    n_clusters = 5
    
    #before performing kmeans:
    # kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    # kmeans.fit(data_scaled)
    # clusterLabel = kmeans.labels_

    #after performing kmeans
    clusterLabel = np.load(idir + "Cluster_Labels.npy")
    d = {
        "Season": Season_ifconcantenated,
        "Coordinates": Coordinates_ifconcantenated,
        'U_diff': data_scaled[:,0], 
        'V_diff': data_scaled[:,2], 
        "Div" : data_scaled[:,1], 
        'Vor': data_scaled[:,11],
        'T_diff': data_scaled[:,3],
        whichMoisture: data_scaled[:,4], 
        'PS': data_scaled[:,5],
        'CAPE': data_scaled[:,6],
        'OMEGAU_Flux': data_scaled[:,7], 
        'OMEGAV_Flux': data_scaled[:,8], 
        'OMEGAT_Flux': data_scaled[:,9],
        'OMEGAQ_Flux': data_scaled[:,10],
        "PRECC": data_PRECC_scaled[:,0], 
        "PRECL":data_PRECL_scaled[:,0],
        'cluster_label': clusterLabel,
        "V_bot": data_scaled[:,12], 
        "T_diff2": data_scaled[:,13], 
    }
    df = pd.DataFrame(data=d)

