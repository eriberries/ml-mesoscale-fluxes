import torch
import math

def BalancedL1Loss(pred, true):
    """
    Balanced L1 Loss used in Libra R-CNN.

    Reference:
        - Pang et al., Few-Shot Object Detection on Remote Sensing Images
        - Libra R-CNN implementation:
          https://github.com/OceanPang/Libra_R-CNN/

    Formula:
        For diff = |pred - true|:

        if diff < 1:
            loss = (a/b) * ( (b*diff + 1) * log(b*diff + 1) - b*diff )
        else:
            loss = c*diff + c/b - a

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor, shape (batch, features)
    true : torch.Tensor
        Target tensor, same shape as pred

    Returns
    -------
    torch.Tensor
        Loss per sample, shape (batch,)
    """

    a = 0.5
    c = 1.5
    b = math.exp(c / a) - 1  

    diff = torch.abs(pred - true)

    loss = torch.where(
        diff < 1, 
        (a / b) * ((b * diff + 1) * torch.log(b * diff + 1) - b * diff),
        c * diff + c / b - a
    )

    return torch.mean(loss, dim=1)

def HuberLoss_c(pred, true, delta=1.0):
    diff = torch.abs(pred-true)
    loss = torch.where(
        diff < delta, 
        0.5*diff**2,
        delta*diff - 0.5*delta**2
    )
    return torch.mean(loss, dim=1)

def L1Loss_c(pred, true):
    diff = torch.abs(pred-true)
    return torch.mean(diff, dim=1)

def MSELoss_c(pred, true):
    diff = torch.abs(pred-true)
    return torch.mean(diff**2, dim=1)

def CustomLoss(pred, true, epsilon=1e-12):
    """
    Custom loss function with multiple experimental variants.
    Only ONE variant is active at a time.
    
    Variants:
    ---------
    (A) Weighted squared error (ACTIVE)
        - Amplifies errors where |true| is small.
        - Treats large positive outliers (diff > 1) differently.
        
    (B) Log-difference loss
        - Works on log(|x| + 1) transform.
        - Mild compression of magnitude.

    (C) Log-difference on quartic transform
        - Stronger penalization of large errors.
        - log(10*x^4 + 1)

    (D) Log-difference on squared transform
        - Moderate compression.
        - log(10*x^2 + 1)

    (E) Signed-log transform loss
        - Retains sign information.
        - log(|x| + 1) multiplied with sign(x)

    (F) Signed-log with stronger scaling
        - Same as (E) but scaled with factor 10 inside log.
    
    To switch variants, comment/uncomment the relevant BLOCK.
    """

    diff = pred - true

    # ---------------------------------------------------------------
    # (A) Weighted squared loss (ACTIVE)
    # ---------------------------------------------------------------
    loss_arr = torch.where(
        diff > 1,
        diff**2 / (torch.abs(true) + epsilon),   
        diff**2                                  
    )
    return torch.mean(loss_arr, dim=1)

    # ---------------------------------------------------------------
    # (B) Log-difference loss (simple)
    # ---------------------------------------------------------------
    # pred_t = torch.log(torch.abs(pred) + 1)
    # true_t = torch.log(torch.abs(true) + 1)
    # return torch.mean((pred_t - true_t)**2, dim=1)

    # ---------------------------------------------------------------
    # (C) Log-difference on quartic values (very strong scaling)
    # ---------------------------------------------------------------
    # pred_t = torch.log(10 * pred**4 + 1)
    # true_t = torch.log(10 * true**4 + 1)
    # return torch.mean((pred_t - true_t)**2, dim=1)

    # ---------------------------------------------------------------
    # (D) Log-difference on squared values (medium scaling)
    # ---------------------------------------------------------------
    # pred_t = torch.log(10 * pred**2 + 1)
    # true_t = torch.log(10 * true**2 + 1)
    # return torch.mean((pred_t - true_t)**2, dim=1)

    # ---------------------------------------------------------------
    # (E) Signed log transform
    # ---------------------------------------------------------------
    # pred_t = torch.sign(pred) * torch.log(torch.abs(pred) + 1)
    # true_t = torch.sign(true) * torch.log(torch.abs(true) + 1)
    # return torch.mean((pred_t - true_t)**2, dim=1)

    # ---------------------------------------------------------------
    # (F) Signed log transform with scaling
    # ---------------------------------------------------------------
    # pred_t = torch.sign(pred) * torch.log(10 * torch.abs(pred) + 1)
    # true_t = torch.sign(true) * torch.log(10 * torch.abs(true) + 1)
    # return torch.mean((pred_t - true_t)**2, dim=1)



def get_loss_function(loss_name):
    """
    Return the appropriate loss function by name.

    Parameters
    ----------
    loss_name : str
        Name of the loss function to return.
        Supported options:
            - "MSE"            : MSELoss_c
            - "MSLE"           : CustomLoss
            - "BalancedL1Loss" : BalancedL1Loss
            - "Huber"          : HuberLoss_c
            - "L1"             : L1Loss_c
            - "Costum"         : CustomLoss

    Returns
    -------
    callable
        A loss function expecting inputs (pred, true) and returning loss.

    Raises
    ------
    ValueError
        If an unknown loss_name is passed.

    """

    loss_dispatch = {
        "MSE": MSELoss_c, # torch.nn.MSELoss()
        # "MSLE": CustomLoss,                  
        "BalancedL1Loss": BalancedL1Loss,
        "Huber": HuberLoss_c,#torch.nn.HuberLoss()
        "L1": L1Loss_c,# torch.nn.L1Loss()
        "Costum": CustomLoss,
    }

    if loss_name not in loss_dispatch:
        raise ValueError(
            f"Unknown loss name '{loss_name}'. "
            f"Available names: {list(loss_dispatch.keys())}"
        )

    return loss_dispatch[loss_name]
