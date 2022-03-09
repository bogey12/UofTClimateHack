from einops import rearrange
import torch
def preprocessing(config, batch_features):
    MEAN = 299.17117
    STD = 146.06215
    mi1 = None
    ma1 = None
    if config['in_opt_flow']:
        batch_features = rearrange(batch_features, 'b t h w c -> b (t c) h w', c=2) 
    if config['convert']:
        batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1)
    if config['normalize']:
        if config['local_norm']:
            MEAN = torch.mean(batch_features)
            STD = torch.std(batch_features)
        if config['normalize'] == 'standardize':
            batch_features -= MEAN
            batch_features /= STD
        elif config['normalize'] == 'minmax':
            mi1 = batch_features.min()
            ma1 = batch_features.max()
            batch_features -= mi1
            batch_features /= (ma1 - mi1)
            batch_features *= 2
            batch_features -= 1
    return batch_features, {'MEAN':MEAN, 'STD':STD, 'mi1':mi1, 'ma1':ma1}

def postprocessing(config, predictions, extra):
    MEAN = extra['MEAN']
    STD = extra['STD']
    mi1 = extra['mi1']
    ma1 = extra['ma1']
    if config['convert']:
        predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
    if config['normalize']:
        if config['output_std']:
            MEAN = config['output_mean']
            STD = config['output_std']
        if config['normalize'] == 'standardize':
            predictions *= STD
            predictions += MEAN
        elif config['normalize'] == 'minmax':
            predictions += 1
            predictions /= 2
            predictions *= (ma1 - mi1)
            predictions += mi1
    return predictions
