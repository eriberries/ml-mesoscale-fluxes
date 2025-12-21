whichMoisture = "Q"
variables = ["U","V","T","PS","OMEGA", whichMoisture, "dU_dx", "dU_dy","dV_dx", "dV_dy", "dT_dx", "dT_dy", "CAPE"]

category0 = ["U","V","T","PS"]
category1 = ["dT_dx", "dT_dy"]
category2 = ["dU_dx", "dV_dy"]
category3 = ["dV_dx", "dU_dy"]
category4 = ["Q", "CAPE"]
category5 = ["OMEGA"]

variables_incl = [category0]  + [category0 + category1, 
                                 category0 + category2,
                                 category0 + category3,
                                 category0 + category4,
                                 category0 + category5] + [category0 + category1 + category2,
                                                           category0 + category1 + category2 + category3,
                                                           category0 + category1 + category2 + category3 + category4,
                                                           category0 + category1 + category2 + category3 + category4 + category5, 
                                                           category0 + category1 + category2 + category3 + category4 + category5] 

def RemoveList(var_incl):
    return [var for var in variables if var not in var_incl]

variables_excl = []
for var_incl in variables_incl:
    variables_excl.append(RemoveList(var_incl))


modelname = ["nn_to_lev_cat0_epoch16", "nn_to_lev_cat01_epoch16","nn_to_lev_cat02_epoch16", 
             "nn_to_lev_cat03_epoch16","nn_to_lev_cat04_epoch16", "nn_to_lev_cat05_epoch16", 
             "nn_to_lev_cat012_epoch16", "nn_to_lev_cat0123_epoch16", "nn_to_lev_cat01234_epoch16",
             "nn_to_lev_cat012345_epoch16", "lev_to_lev_cat012345_epoch16"]

testfile = []

cat_keys = ["cat0", 
           "cat01",
           "cat02", 
           "cat03", 
           "cat04", 
           "cat05",
           "cat012",
           "cat0123",
           "cat01234",
           "nn_to_lev", 
           "lev_to_lev"]

experiment_info = {}
for i, key in enumerate(cat_keys):
    experiment_info[key] = {
        "var_incl": variables_incl[i],
        "var_excl": variables_excl[i],
        "modelname": modelname[i]
    }

for i, key in enumerate(experiment_info.keys()):
    print(key, experiment_info[key])