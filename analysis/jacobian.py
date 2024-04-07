"""
Modified from https://github.com/LabForComputationalVision/memorization_generalization_in_diffusion_models
"""

import numpy as np
import torch


def calc_jacobian(inp, prefs, model, layer_num=None, channel_num=None):
    assert inp.shape[0] == 1 # only for 1 sample per run
    ############## prepare the static model
    # model = Net(all_params)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    ############## find Jacobian
    if inp.requires_grad is False:
        inp.requires_grad = True

    cond = None
    time = torch.zeros(1).to(inp.device, dtype=torch.float32)
    returns = torch.multiply(torch.ones_like(prefs).to(inp.device, dtype=torch.float32), prefs) * 7 / 1000
    out = model(inp, cond, time, prefs, returns, use_dropout=False)
    jacob = []

    # start_time_total = time.time()
    for i in range(inp.size()[1]):
        for j in range(inp.size()[2]):
            part_der = torch.autograd.grad(
                out[0, i, j], inp, retain_graph=True
            )
            jacob.append(part_der[0][0].data.reshape(-1))
    # print("----total time to compute jacobian --- %s seconds ---" % (time.time() - start_time_total))

    return torch.stack(jacob)
