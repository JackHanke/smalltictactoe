import matplotlib.pyplot as plt
import torch

def plot_weights(
        model,
        ispruned: bool = False
    ):

    # fig = plt.figure(figsize=(8,8))

    if ispruned:
        arr = torch.cat([model.model[0].weight.detach().transpose(0,1), model.model[2].weight.detach()])
    else:
        arr = torch.cat([model.model[0].weight.detach().transpose(0,1), model.model[2].weight.detach()])

    plt.imshow(arr, cmap='magma')
    plt.colorbar()
    plt.title(f'Weights of network')
    plt.show()



