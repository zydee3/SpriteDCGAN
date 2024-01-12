from model.adversary import Adversary, AdversaryType
from torch.nn import Linear, Parameter, Module
from torch import flatten, einsum, abs, sum, exp, cat, randn
from torch.cuda.amp import autocast


class MiniBatchDiscriminator(Module):
    def __init__(self, in_features, out_features, inter_features):
        super(MiniBatchDiscriminator, self).__init__()

        if out_features <= in_features:
            raise Exception("out_features must be greater than in_features")

        self.T = Parameter(randn(
            in_features,
            out_features - in_features,
            inter_features,
        ))


    def forward(self, tensor):
        # Flattening the input: Convert 2D feature maps to 1D feature vectors
        # Input shape: [N, Channels, Height, Width]
        # Output shape: [N, Channels*Height*Width]
        flattened_input = flatten(tensor, start_dim=1)

        # Minibatch feature calculation using tensor product
        # 'self.T' is a trainable parameter tensor that introduces inter-sample relationships
        # Output shape: [N, Out, Inter]
        minibatch_features = einsum("ni,ijk->njk", flattened_input, self.T)

        # Calculating differences between samples in the minibatch
        # Broadcasting the subtraction across all combinations of samples in the minibatch
        # Output shape: [N, N, Out, Inter]
        minibatch_diff = abs(minibatch_features.unsqueeze(dim=1) - minibatch_features)

        # Summing differences (L1-norm) across the minibatch
        # Output shape: [N, N, Out]
        l1_norm_diff = sum(minibatch_diff, dim=3)

        # Applying the exponential function to the negative L1-norm
        # This encodes how different each pair of samples in the minibatch is
        # Output shape: [N, N, Out]
        exp_l1_norm_diff = exp(-l1_norm_diff)

        # Summing across batches to get a vector encoding the diversity of the batch
        # Output shape: [N, Out]
        diversity_vector = sum(exp_l1_norm_diff, dim=1)

        # Concatenating the original flattened input with the diversity vector
        # This forms the enhanced feature vector for the subsequent layers
        # Output shape: [N, Channels*Height*Width + Out]
        enhanced_features = cat((flattened_input, diversity_vector), dim=1)

        return enhanced_features


class Discriminator(Adversary):
    def __init__(self, noise_size, image_size, channel_size, base_width):
        super().__init__(AdversaryType.Discriminator, noise_size, image_size, channel_size, base_width)
        
        
    def __repr__(self):
        MB = self.layers[-2]
        return f"Discriminator = {super().__repr__()}\nMinibatchDisc(in_features={MB.T.shape[0]}, out_features={MB.T.shape[1]})"
        
        
    def add_linear_layer(self, in_factor): 
        layer_id = len(self.layers)
        layer_name = self._generate_layer_name(Linear, layer_id)
        
        output_size = self.base_width * in_factor * self.final_size * self.final_size
        layer = Linear(output_size, 1)
        
        self.layers.add_module(layer_name, layer)
    
    
    def add_mini_batch_layer(self, in_factor, out_factor, inter_value):
        layer_id = len(self.layers)
        layer_name = self._generate_layer_name(MiniBatchDiscriminator, layer_id)
        
        layer = MiniBatchDiscriminator(
            self.base_width * in_factor * self.final_size * self.final_size,
            self.base_width * out_factor * self.final_size * self.final_size,
            inter_value
        )
        
        self.layers.add_module(layer_name, layer)