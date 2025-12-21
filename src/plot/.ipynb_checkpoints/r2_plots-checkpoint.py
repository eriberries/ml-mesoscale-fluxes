def Plot_r2_profile(
    r2_arr, 
    compare = False, 
    plot_variance= False, 
    variance_arr = None, 
    plot_MSE= False, 
    MSE_arr = None, 
    title_add = "", 
    xlim=(0,1)
):
    
    variables = ['OMEGAQ_Flux', 'OMEGAT_Flux', 'OMEGAU_Flux', 'OMEGAV_Flux']
    tick_labels = [f"{levs[::-1][j]:.1f}" for j in plevticks][::-1]
    
    n, m = 2,2 
    fig, ax = plt.subplots(n, m, figsize=(10,8))
    axes = [ax[i][j] for j in range(m) for i in range(n)]
    
    for i in range(4):
        ax_i = axes[i]
        if compare:
            alpha=0.5
        else:
            alpha=1
        if type(r2_arr) == list:
            for j in range(len(r2_arr)):
                arr_j = r2_arr[j]
                r2_labels = ["R², MLR", "R², ANN"]
                ax_i.barh(
                    np.arange(n_levs), 
                    arr_j[n_levs*i:n_levs*(i+1)], 
                    alpha=alpha, 
                    label = r2_labels[j]
                )
        else:
            ax_i.barh(
                np.arange(n_levs), 
                r2_arr[n_levs*i:n_levs*(i+1)], 
                color='skyblue', 
                alpha=alpha
            )
        
        if plot_variance:
            ax2 = ax_i.twiny()
            
            if type(variance_arr) == list:
                ax2.plot(
                    variance_arr[0][n_levs*i:n_levs*(i+1)]**.5, 
                    np.arange(n_levs), 
                    color='black', 
                    label="Std. Dev."
                )
            else:
                ax2.plot(
                    variance_arr[n_levs*i:n_levs*(i+1)]**.5, 
                    np.arange(n_levs), 
                    color='black', 
                    label="Std. Dev."
                )
            if type(MSE_arr) == list: 
                if plot_MSE:
                    ax2.plot(
                        MSE_arr[0][n_levs*i:n_levs*(i+1)]**.5, 
                        np.arange(n_levs), 
                        "--", 
                        color="blue", 
                        label="RMSE, MLR"
                    )
                    ax2.plot(
                        MSE_arr[1][n_levs*i:n_levs*(i+1)]**.5, 
                        np.arange(n_levs), 
                        color="blue", 
                        label="RMSE, ANN"
                    )
            else:
                if plot_MSE:
                    ax2.plot(
                        MSE_arr[n_levs*i:n_levs*(i+1)]**.5, 
                        np.arange(n_levs), 
                        color="blue", 
                        label="RMSE"
                    )

            ax2.set_xlabel('std') 
            ax2.set_xlim(0,7)

        ax_i.invert_yaxis()
        ax_i.set_yticks(ticks=plevticks, labels=tick_labels)
        ax_i.grid(axis='x')
        ax_i.set_xlim(xlim[0], xlim[1])
        
        ax_i.set_xlabel("R² Value")
        ax_i.set_ylabel("Pressure (linear scale) [hPa]")

        ax_i.set_title(variables[i] + title_add)
        
    handles, labels = ax_i.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='upper right', 
        bbox_to_anchor=(1.01, 0.91), 
        borderaxespad=1
    )
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='upper right', 
        bbox_to_anchor=(1.01, 0.85), 
        borderaxespad=1
    )
    fig.tight_layout()
    return fig

def Plot_r2_map(r2_arr, title_add = "", quantity = "r2", clev_range = (-1,1)):
    
    variables = ['OMEGAQ_Flux', 'OMEGAT_Flux', 'OMEGAU_Flux', 'OMEGAV_Flux']
    clev = np.linspace(clev_range[0], clev_range[1], 81)
    
    n, m = 2,2 
    fig, ax = plt.subplots(
        n, m, figsize=(12,8), 
        subplot_kw={'projection':ccrs.PlateCarree(central_longitude=0)}
    )
    axes = [ax[i][j] for j in range(m) for i in range(n)]

    lons_hat = lons[1:-1]
    lats_hat = lats
    
    for i in range(4):
        data_arr = r2_arr[i].T
        ax_i = axes[i]

        cyclic_data, cyclic_lon = add_cyclic_point(data_arr, coord=lons_hat)
        plot = ax_i.contourf(
            cyclic_lon, lats_hat, cyclic_data, 
            transform=ccrs.PlateCarree(), 
            cmap = "cubehelix", levels=clev, extend="both"
        )

        ax_i.coastlines(color='k')
        gl = ax_i.gridlines(
            crs=ccrs.PlateCarree(central_longitude=180.0), 
            draw_labels=True, linewidth=.6, color="gray", 
            alpha=0.5, linestyle='-.'
        )
        gl.xlabel_style = {"size" : 7}
        gl.ylabel_style = {"size" : 7}
        ax_i.set_title(variables[i])
        
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    cbar = fig.colorbar(
        plot, shrink=0.5, ax=ax, location='bottom'
    ).set_label(label = "r^2 score", size=12)
        
    return fig