optimizer_name = "Adam"
cycle_momentum = True 
cyclic_mode='triangular2'
criterion = "BalancedL1Loss"
L1reg = False
l1_reg_strength = None
isauxiliary = False

mask_14Qlevels = True
mask = torch.ones(out_features, dtype=torch.float32)
mask[:4] = 0

