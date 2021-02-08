import numpy as np
import torch


def get_smape(targets, preds):

    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()

    # percentage error, zero if both true and pred are zero
    numerator = np.abs(targets - preds)
    denominator = (np.abs(targets) + np.abs(preds))
    daily_errors_arr = 200 * np.nan_to_num(numerator / denominator)
    signed_err = preds - targets
    return daily_errors_arr, signed_err
