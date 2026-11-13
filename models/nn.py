import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundToInt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale=1.0):
        # Scale weights so they span across integer steps, then round
        scaled_weight = weight * scale
        rounded_weight = torch.round(scaled_weight)
        # Scale back down for the forward pass (or keep unscaled depending on your architecture)
        return rounded_weight / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients straight through unmodified
        return grad_output, None

class IntegerWeightLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=10.0):
        super().__init__(in_features, out_features, bias)
        self.scale = scale  # Controls the resolution/range of your integer weights
        
    def forward(self, input):
        # Quantize weights for the forward pass
        w_quantized = RoundToInt.apply(self.weight, self.scale)
        b_quantized = RoundToInt.apply(self.bias, self.scale)
        return F.linear(input, w_quantized, b_quantized)

class tttclassifer(nn.Module):
    def __init__(self,
        hidden_sizes: list[int],
        input_size: int = 9,
        output_size: int = 2,
    ):
        super(tttclassifer, self).__init__()

        layers = []
        self.hidden_sizes = hidden_sizes
        hidden_sizes = [input_size] + hidden_sizes + [output_size]
        self.hidden_sizes = hidden_sizes

        for i, o in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(IntegerWeightLinear(i, o))
            layers.append(nn.ReLU()) # Add activation between layers
        # 3. Remove the last ReLU (usually not needed on the output layer)
        layers.pop()
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y



class TicTacToeNet(nn.Module):
    def __init__(self,
            hidden_sizes: list[int],
            input_size: int = 18,
            do_illegal_move_masking: bool = True
        ):
        super(TicTacToeNet, self).__init__()
        self.do_illegal_move_masking = do_illegal_move_masking
        self.input_size = input_size
        
        layers = []
        if hidden_sizes[0] is None:
            self.hidden_sizes = [hidden_sizes, 9]
            layers.append(nn.Linear(input_size, 9))
        else:
            self.hidden_sizes = hidden_sizes
            hidden_sizes = [input_size] + hidden_sizes + [9]
            self.hidden_sizes = hidden_sizes

            for i, o in zip(hidden_sizes[:-1], hidden_sizes[1:]):
                layers.append(nn.Linear(i, o))
                layers.append(nn.ReLU()) # Add activation between layers
            # 3. Remove the last ReLU (usually not needed on the output layer)
            layers.pop()
        
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        y = self.model(x)

        if self.do_illegal_move_masking:
            # mask_val = float('-inf')
            mask_val = -1e6
            if self.input_size == 9 or self.input_size == 17 or self.input_size == 33:
                illegal = (x[:, :9] != 0)
                y = y.masked_fill(illegal, mask_val)
            elif self.input_size == 18:
                illegal = torch.bitwise_or((x[:, :9] != 0), (x[:, 9:] != 0))
                y = y.masked_fill(illegal, mask_val)
            else:
                raise Exception('Something went wrong with illegal move masking.')
        return y


class routerNet(nn.Module):
    def __init__(self, hidden_sizes: list[int], input_size: int = 9):
        super(routerNet, self).__init__()
        
        output = 2

        layers = []
        if hidden_sizes[0] is None:
            self.hidden_sizes = [hidden_sizes, output]
            layers.append(nn.Linear(input_size, output))
        else:
            self.hidden_sizes = hidden_sizes
            hidden_sizes = [input_size] + hidden_sizes + [output]
            self.hidden_sizes = hidden_sizes

            for i, o in zip(hidden_sizes[:-1], hidden_sizes[1:]):
                layers.append(nn.Linear(i, o))
                layers.append(nn.ReLU()) # Add activation between layers
            # 3. Remove the last ReLU (usually not needed on the output layer)
            layers.pop()
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        return logits
    


# class simpleNet(nn.Module):
#     def __init__(self, size: int):
#         super(simpleNet, self).__init__()
#         self.size = size
#         self.model = nn.Sequential(
#             nn.Linear(self.size, self.size),
#             nn.SiLU(),
#             nn.Linear(self.size, 2),
#         )
#     def forward(self, x):
#         x = self.model(x)
#         return x