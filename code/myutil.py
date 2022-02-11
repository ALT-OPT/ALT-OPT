def sort_trials(trials, key=None):
    if key is None:
        return trials
    # import ipdb; ipdb.set_trace()
    if not key in trials[0].params.keys():
        return trials
        
    def get_key(trial):
        return trial.params[key]
    trials.sort(key=get_key)
    return trials

def set_signal_by_label(x, data):
    mask = data.train_mask
    y = data.y
    x[:,:] = 0
    x[mask, data.y[mask]] = 1
    return x
