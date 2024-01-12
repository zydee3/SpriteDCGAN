from torch.nn import init, Module, Sequential, Conv2d, ReLU, BatchNorm2d, ConvTranspose2d, Tanh, LeakyReLU, Sigmoid, Dropout, Linear
from enum import auto, Enum
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


class AdversaryType(Enum):
    Generator = auto()
    Discriminator = auto()
    

class Adversary(Module):
    def __init__(self, adversary_type, noise_size, image_size, channel_size, base_width):
        
        super().__init__()
        
        if image_size % 16 != 0:
            raise Exception("Image size must be a multiple of 16.")
        
        self.adversary_type = adversary_type
        self.noise_size = noise_size
        self.image_size = image_size
        self.final_size = (image_size // 16)
        self.channel_size = channel_size
        self.base_width = base_width    
            
        self.layers = Sequential()

        self.scheduler = None
        self.optimizer = None
        self.loss_function = None
        
        self.is_built = False
        
    
    def __repr__(self):  
        output = "Sequential(\n"   
        
        start_idx = 0
        stop_idx = len(self.layers)
        
        # Add the linear layer if it exists for the generator
        if self.adversary_type == AdversaryType.Generator and self.linear_layer is not None:
            output += f"\t{self.linear_layer} (shaped out_features={self.linear_layer.out_features // (self.final_size ** 2)})\n"
            start_idx = 1
            stop_idx += 1
            
        for idx, layer in enumerate(self.layers, start=start_idx):
            
            # Add a new line before each block for readability as long 
            # as it's not the first layer
            if (layer.__class__.__name__ == "ConvTranspose2d" or layer.__class__.__name__ == "Conv2d") and idx > 0:
                output += "\n"
                
            if idx == stop_idx - 2 and self.adversary_type == AdversaryType.Discriminator:
                if self.layers[idx].__class__.__name__ == "Linear":
                    output += "\n"
            
            # Add a new line before the output layer for readability
            if idx == stop_idx - 1:
                output += "\n"
               
            output += f"\t{layer}\n"
        
        output += ")"
        
        return output

    
    def get_learning_rate(self):
        return self.scheduler.get_last_lr()[0]

        
    def initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0)
    

    def _generate_layer_name(self, module, layer_id):
        # return f"{module.__class__.__name__.lower()}_{layer_id}" 
        return f"{layer_id}"

    
    def add_convolutional_layer(self, in_size=None, out_size=None, in_factor=None, out_factor=None, kernel_size=4, stride=2, padding=1, use_bias=False):
        assert (in_size is not None and out_size is not None) or (in_factor is not None and out_factor is not None)
        
        layer_id = len(self.layers)
        layer_name = self._generate_layer_name(ConvTranspose2d, layer_id)
        
        if in_size is None:
            in_size = self.base_width * in_factor
            out_size = self.base_width * out_factor
        
        layer = None
        
        if self.adversary_type == AdversaryType.Generator:
            layer = ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        else:
            layer = Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.layers.add_module(layer_name, layer)
        
    
    def _add_batch_norm_layer(self, out_factor):
        layer_id = len(self.layers)
        layer_name = self._generate_layer_name(BatchNorm2d, layer_id)
        
        out_size = self.base_width * out_factor
        
        layer = BatchNorm2d(out_size)
        self.layers.add_module(layer_name, layer)
    
    
    def add_activator_layer(self):
        layer_id = len(self.layers)
        layer_name = self._generate_layer_name(ReLU, layer_id)
        
        layer = None
        
        if self.adversary_type == AdversaryType.Generator:
            layer = ReLU(inplace=True)
        else:
            layer = LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.layers.add_module(layer_name, layer)
    
    
    def _add_dropout_layer(self, dropout_rate):
        layer_id = len(self.layers)
        layer_name = self._generate_layer_name(Dropout, layer_id)
        layer = Dropout(p=dropout_rate)
        self.layers.add_module(layer_name, layer)
    
    
    def add_block(self, in_factor, out_factor, use_batch_norm=True, use_activator=True, use_bias=False, kernel_size=4, stride=2, padding=1, dropout_rate=0.0):
        assert use_bias == (not use_batch_norm)
        
        if self.is_built == True:
            raise Exception("Attempting to add new weights to an already built model.")
        
        self.add_convolutional_layer(in_factor=in_factor, out_factor=out_factor, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=use_bias)
        
        if use_batch_norm:
            self._add_batch_norm_layer(out_factor)
        
        if use_activator:
            self.add_activator_layer()
        
        if dropout_rate > 0.0:
            self._add_dropout_layer(dropout_rate)
 
    
    def build(self):
        layer_id = len(self.layers)
        layer = None
        
        if self.adversary_type == AdversaryType.Generator:
            layer_name = self._generate_layer_name(Tanh, layer_id)
            layer = Tanh()
            self.layers.add_module(layer_name, layer)

        self.apply(self.initialize_weights)
        self.is_built = True
    
    
    def forward(self, input):
        if self.is_built == False:
            raise Exception("Attempting to forward step without initializing the model. Please use model.initialize().")
        
        return self.layers(input)


    def set_optimizer(self, optimizer=AdamW, learning_rate=0.0002, beta1=0.5, beta2=0.999):        
        self.optimizer = optimizer(self.parameters(), lr=learning_rate, betas=(beta1, beta2))
    
    
    def set_scheduler(self, scheduler=CosineAnnealingLR, lr_reset_epoch_rate=2, min_lr=0.0001):
        if self.optimizer is None:
            raise Exception("Attempting to set a scheduler without an optimizer. Please use model.set_optimizer().")
        
        self.scheduler = scheduler(self.optimizer, T_max=lr_reset_epoch_rate, eta_min=min_lr)