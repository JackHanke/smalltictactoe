import matplotlib.pyplot as plt
import torch

def plot_weights(
        model,
        ispruned: bool = False,
        title: str = '',
    ):

    # fig = plt.figure(figsize=(,8))
    fig, (ax1) = plt.subplots(1, 1, figsize=(9, 9))  # a figure with a 2x2 grid of Axes

    # parameters_to_tune = []
    # for i in range(len(model.hidden_sizes)-1):
    #     parameters_to_tune.append((model.model[2*i], 'weight'))
    #     parameters_to_tune.append((model.model[2*i], 'bias'))
    arr = []
    for i in range(len(model.hidden_sizes)-1):
        if i == 0:
            weight_matrix = model.model[2*i].weight.detach().transpose(0,1)
        else:
            weight_matrix = model.model[2*i].weight.detach()
        arr.append(weight_matrix)
    arr = torch.cat(arr).transpose(0,1)

    arr = arr.masked_fill(arr == 0, torch.nan)


    ax1.imshow(arr, cmap='magma')
    ax1.set_title(f'{title}')
    # ax1.colorbar()
    # plt.show()
    # ax2.imshow(arr == 0, cmap='magma')
    # ax2.set_title(f'Location of 0 weights')
