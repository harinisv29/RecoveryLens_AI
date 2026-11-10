import torch

def integrated_gradients(model,x,mask,steps=32):
    # Baseline: training mean physiology; preserve observation indicators.
    baseline=x.clone(); baseline[:,:,:7]=0
    difference=x-baseline; gradients=torch.zeros_like(x)
    for i in range(steps+1):
        z=(baseline+difference*(i/steps)).detach().requires_grad_(True)
        output=model(z,mask).sum()
        gradient=torch.autograd.grad(output,z)[0]
        gradients+=gradient*(.5 if i in (0,steps) else 1)
    attribution=(difference*gradients/steps)[0,:,:7].detach().cpu()
    return {'feature_attribution':attribution.abs().sum(0).tolist(),'temporal_heatmap':attribution.tolist(),'important_intervals':torch.topk(attribution.abs().sum(1),min(5,len(attribution))).indices.tolist(),'method':'Integrated Gradients of logit; training-mean baseline, fixed missingness; not causal','units':'logit attribution'}
