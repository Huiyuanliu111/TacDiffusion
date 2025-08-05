import torch
import numpy as np

def temporal_aggregation(i, all_actions, all_time_actions, num_queries, k):
    """
    Performs temporal aggregation of actions using exponential weighting.
    """
    end_time_idx = min(i + num_queries, all_time_actions.shape[1])
    seq_length = min(num_queries, end_time_idx - i)
    all_time_actions[i, i:i + seq_length] = all_actions[:seq_length]
    actions_for_curr_step = all_time_actions[:i + 1, i]
    actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    if len(actions_for_curr_step) > 0:
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    else:
        raw_action = all_actions[0:1]

    return raw_action