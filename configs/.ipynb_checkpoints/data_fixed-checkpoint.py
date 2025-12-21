X = 5
norm_in = "level" 
norm_out = "momentum"  
whichMoisture = "Q" 
discard10 = True 
discard10_str = "discard10" if discard10 else "keep10"
isextremes = True 
extremes_str = "withExtremes" if isextremes else "noExtremes"
extremes_th = 35
extremes_str += f"{extremes_th}" if extremes_str == "noExtremes" else ""
ismemory = False
ismemory_str = "withMemory" if ismemory else "noMemory"

string_for_filename = f"_{whichMoisture}_{discard10_str}_{extremes_str}_{ismemory_str}"

n_levs = 22 if discard10 else 32
out_features = 4*n_levs
in_features = 2*(11*n_levs + 2) if ismemory else  11*n_levs + 2 

N_time = 7320
N_lat = 29 
N_lev = 22 if discard10 else 32
N_lon = 47