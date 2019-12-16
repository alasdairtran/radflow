import numpy as np


def get_smape(target_tensors, pred_tensors):

    targets = target_tensors.cpu().numpy()
    preds = pred_tensors.cpu().numpy()

    # percentage error, zero if both true and pred are zero
    numerator = np.abs(targets - preds)
    denominator = (np.abs(targets) + np.abs(preds))
    daily_smape_arr = 200 * np.nan_to_num(numerator / denominator)
    return np.mean(daily_smape_arr, axis=1).tolist(), daily_smape_arr.tolist()
