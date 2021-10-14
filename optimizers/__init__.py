import torch 
from . import sps


def get_optimizer(opt_dict, model, exp_dict):
    name = opt_dict['name']
    
    if name == "adam":
        opt = torch.optim.Adam(
                model.parameters(), lr=opt_dict["lr"], betas=(0.99, 0.999))

    elif name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=opt_dict["lr"])

    elif name == "sps":
        opt = sps.Sps(
            model.parameters(), c=opt_dict['c'], momentum=opt_dict.get('momentum', 0.))
    return opt