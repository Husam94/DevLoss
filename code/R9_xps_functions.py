import numpy as np
import math

#------------------------------------

import torch
import torch.nn as nn

#------------------------------------

import poseigen_seaside.basics as se
import poseigen_trident.utils as tu

#===========================================================================

def shuffle_train_val_y(data_split, Y, has_val=True, random_state=None):
    """
    Shuffle Y for train and (optionally) validation sets only.
    
    Args:
        data_split: list of lists of indices [train, val, test, ...]
        Y: np.ndarray or list, shape (n_obs, ...)
        has_val: bool, if True, shuffle validation set as well
        random_state: int or None, for reproducibility
        
    Returns:
        Y_shuf: same type/shape as Y, with train/val shuffled
    """
    rng = np.random.default_rng(random_state)
    Y_shuf = np.array(Y).copy()  # works for list or np.ndarray

    # Shuffle train
    train_idx = data_split[0]
    shuf_train = rng.permutation(Y_shuf[train_idx])
    Y_shuf[train_idx] = shuf_train

    # Shuffle val if present and requested
    if has_val and len(data_split) > 1:
        val_idx = data_split[1]
        shuf_val = rng.permutation(Y_shuf[val_idx])
        Y_shuf[val_idx] = shuf_val

    # If Y was a list, convert back
    if isinstance(Y, list):
        Y_shuf = list(Y_shuf)
    return Y_shuf

# # --- Test Example with Validation Set ---
# Y = np.arange(100)
# idxs = np.arange(100)
# rng = np.random.default_rng(42)
# rng.shuffle(idxs)
# train_idx = idxs[:60]
# val_idx = idxs[60:80]
# test_idx = idxs[80:]

# data_split = [train_idx, val_idx, test_idx]

# # Shuffle train and val
# Y_shuf = shuffle_train_val_y(data_split, Y, has_val=True, random_state=123)

# print("Original Y (train):", Y[train_idx][:10])
# print("Shuffled Y (train):", np.array(Y_shuf)[train_idx][:10])
# print("Original Y (val):", Y[val_idx][:5])
# print("Shuffled Y (val):", np.array(Y_shuf)[val_idx][:5])
# print("Original Y (test):", Y[test_idx][:5])
# print("Shuffled Y (test):", np.array(Y_shuf)[test_idx][:5])  # should be unchanged





def Adder(inp, add, scalefactor = 1):
    return (inp * scalefactor) + add
def AntiLog(inp, added, add = 0, scalefactor = 1): 
    return (math.e ** (inp * scalefactor)) - added + add










#=====================================================

def Log_AError(inp1, inp2, weights = None, mean = True,
               eps = 1e-10, pyt = False,            
               rel = False, 
               expo = 2, root = False, weightbefore = False): 
    
    if expo != 2: root = False
    w = se.WeightsAdjuster(inp1, weights)

    rr = inp2 if rel else 1

    if eps is None: eps = 0

    logf = torch.log if pyt else np.log

    e = abs((logf(inp1 + eps) - logf(inp2 + eps))/ rr) #RELATIVE ERROR 

    if weightbefore: w = w **expo
    
    AE = w * (e ** expo)
    
    if mean == True:
        if isinstance(w, int) == False: AE = AE.sum() / w.sum()
        else: AE = AE.mean()

        if root: AE = AE**(1/2)

    return AE


def PearsonError(inp1, inp2, weights = None,
                 pyt = False): 
    
    #Currently only works for single outputs!!!!!!!!!!!!!!!!!!!!!!!!
    
    meanf = torch.mean if pyt else np.mean
    cosf = nn.CosineSimilarity(dim=1, eps=1e-6) if pyt else se.CosineSimilarity
    sumf = torch.sum if pyt else np.sum

    mx = meanf(inp2)
    my = meanf(inp1)

    xm, ym = inp2 - mx, inp1 - my
    
    return sumf(1-cosf(xm,ym))


def Z_AError(inp1, inp2, weights = None, mean = True,
             meano = 0, stdo = 1, 
             rel = False, 
        
             expo = 2, root = False, weightbefore = False): 
    
    if expo != 2: root = False
    w = se.WeightsAdjuster(inp1, weights)

    rr = inp2 if rel else 1

    inp1z, inp2_z = [(x-meano)/stdo for x in [inp1, inp2]]

    e = abs((inp1z-inp2_z) / rr) #RELATIVE ERROR 

    if weightbefore: w = w **expo
    
    AE = w * (e ** expo)
    
    if mean == True:
        if isinstance(w, int) == False: AE = AE.sum() / w.sum()
        else: AE = AE.mean()

        if root: AE = AE**(1/2)

    return AE


def Poisson_NLL(inp1, inp2, weights = None, 
                pyt = False,             
                eps = 1e-6): 

    logf = torch.log if pyt else np.log
    meanf = torch.mean if pyt else np.mean

    return meanf(inp1 - (inp2 * (logf(inp1 + eps))))

def BetaPrime_NLL(inp1, inp2, std = 1, 
                  weights = None, 
                  pyt = False, eps = 1e-10):
    
    logf = torch.log if pyt else np.log
    meanf = torch.mean if pyt else np.mean
    bpf = tu.BetaPrime_PDF_pyt if pyt else tu.BetaPrime_PDF_sci

    alpha1 = std[:, :, :, :, 0]
    alpha2 = std[:, :, :, :, 1]

    lix = bpf(inp1, alpha1, alpha2, eps = eps)
    
    return 0 - meanf(logf(lix + eps))

def BetaPrime_NMP(inp1, inp2, std = 1, 
                  weights = None, 
                  pyt = False, eps = 1e-10):
    
    meanf = torch.mean if pyt else np.mean
    bpf = tu.BetaPrime_PDF_pyt if pyt else tu.BetaPrime_PDF_sci

    alpha1 = std[:, :, :, :, 0]
    alpha2 = std[:, :, :, :, 1]

    lix = bpf(inp1, alpha1, alpha2, eps = eps)
    
    return 0 - meanf(lix)



def get_loss_mode(lsf):
    """
    Returns (lmo, mmo) for the given loss string.
    lmo: PyTorch-friendly (for fitting)
    mmo: numpy-friendly (for evaluation/ensemble selection)
    """
    if lsf in ['MSE', 'Random_MSE']:
        rooto = False
        argos = {'expo': 2, 'root': rooto}
        lmo = [se.AError, {**argos}]
        mmo = [se.AError, {**argos}]
    
    if lsf in ['MAE', 'Random_MAE']:
        rooto = False
        argos = {'expo': 1, 'root': rooto}
        lmo = [se.AError, {**argos}]
        mmo = [se.AError, {**argos}]

    if lsf in ['MSLE-1', 'MSLE-eps']:
        psu = 1 if lsf == 'MSLE-1' else 1e-10
        shard = {'expo': 2, 'root': False, 'eps': psu}
        lmo = [Log_AError, {'pyt': True, **shard}]
        mmo = [Log_AError, {'pyt': False, **shard}]

    if lsf == 'P-NLL':
        lmo = [Poisson_NLL, {'pyt': True, 'eps': 1e-10}]
        mmo = [Poisson_NLL, {'pyt': False, 'eps': 1e-10}]
    if lsf == 'NLL':
        lmo = [BetaPrime_NLL, {'pyt': True, 'eps': 1e-10}]
        mmo = [BetaPrime_NLL, {'pyt': False, 'eps': 1e-10}]

    if lsf == 'NMP':
        lmo = [BetaPrime_NMP, {'pyt': True, 'eps': 1e-10}]
        mmo = [BetaPrime_NMP, {'pyt': False, 'eps': 1e-10}]
    
    if lsf == 'MDE':
        shard = {'expo': 1, 'root': False, 'pseudo': 1e-10}
        lmo = [tu.DeviaError, {'pyt': True, **shard}]
        mmo = [tu.DeviaError, {'pyt': False, **shard}]
    
    if lsf == 'MUDE':
        shard = {'expo': 1, 'root': False, 'pseudo': 1e-10, 'comprel_mode': [tu.C_BetaPrime, {}]}
        lmo = [tu.DeviaError, {'pyt': True, **shard}]
        mmo = [tu.DeviaError, {'pyt': False, **shard}]
    
    return lmo, mmo