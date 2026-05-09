import matplotlib.pyplot as plt
import torch

def plot_weights(
        model,
        ispruned: bool = False
    ):

    # fig = plt.figure(figsize=(,8))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9))  # a figure with a 2x2 grid of Axes

    weight_matrix_1 = model.model[0].weight.detach().transpose(0,1)
    weight_matrix_2 = model.model[2].weight.detach()
    arr = torch.cat([weight_matrix_1, weight_matrix_2])

    ax1.imshow(arr, cmap='magma')
    ax1.set_title(f'Weights of network, 1:{tuple(weight_matrix_1.shape)}, 2:{tuple(weight_matrix_2.shape)}')
    # ax1.colorbar()
    # plt.show()
    ax2.imshow(arr == 0, cmap='magma')
    ax2.set_title(f'Location of 0 weights')
