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

def plot_single_layer_net(
        model,
        title: str = '',
    ):
    fig, (ax1) = plt.subplots(1, 1, figsize=(9, 9))  # a figure with a 2x2 grid of Axes
    
    weight_matrix_0 = model.model[0].weight.detach().transpose(0,1)
    print(weight_matrix_0.shape)
    bias_0 = model.model[0].bias.detach().unsqueeze(0)
    print(bias_0.shape)
    padding = torch.zeros(bias_0.shape)
    weight_matrix_1 = model.model[2*1].weight.detach().transpose(0,1)
    print(weight_matrix_1.shape)
    bias_1 = model.model[2*1].bias.detach().unsqueeze(0)
    print(bias_1.shape)
    print(padding.shape)

    bottom_padding = torch.zeros(bias_0.shape).transpose(1,0)
    print(bottom_padding.shape)

    # weight_matrix_1 = torch.cat(
    #     [weight_matrix_1, bottom_padding]
    # )
    # print(weight_matrix_1.shape)

    arr = torch.cat(
        [
            bias_1,
            padding,
            weight_matrix_1,
            padding,
            padding,
            bias_0,
            padding,
            weight_matrix_0,
        ]
    ).transpose(0,1)

    arr = arr.masked_fill(arr == 0, torch.nan)

    ax1.imshow(arr, cmap='magma')
    ax1.set_title(f'{title}')
