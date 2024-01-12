from enum import Enum, auto
import matplotlib.pyplot as plt
from numpy import linspace, zeros


class LogFlag(Enum):
    LOG_GEN_LOSS = auto()
    LOG_DIS_LOSS = auto()
    LOG_GEN_LEARNING_RATE = auto()
    LOG_DIS_LEARNING_RATE = auto()


class LogValue:
    def __init__(self, num_epochs):
        self.total_values = zeros(num_epochs)
        self.previous_value = 0
        self.current_value = 0
        
    def update(self, idx, value):
        self.total_values[idx] += value        
        self.previous_value = self.current_value
        self.current_value = value
    
    def get_delta(self):            
        return self.current_value - self.previous_value

    
class TrainingLog:
    def __init__(self, num_epochs, num_batches, in_flags: list = {}):  
        self.max_epoch = num_epochs       
        self.max_batches = num_batches       
        self.values = [None] * len(LogFlag)
        
        for flag in LogFlag:
            if (flag not in in_flags) or (in_flags[flag] is True):
                self.values[flag.value - 1] = LogValue(num_epochs)
                
    
    def get_delta(self, flag):
        return self.values[flag.value - 1].get_delta()
    
    
    def get_total(self, flag):
        return self.values[flag.value - 1].total_values            
    
    
    def add_entry(self, epoch, generator_lr=None, discriminator_lr=None, generator_loss=None, discriminator_loss=None):
        for flag_idx, value in enumerate([generator_loss, discriminator_loss, generator_lr, discriminator_lr]):
            if value is not None:
                self.values[flag_idx].update(epoch, value)
                            
    
    def plot(self, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Results")
        
        x = linspace(0, self.max_epoch, self.max_epoch)
        for flag, value in zip(LogFlag, self.values):
            if value is not None:
                y = value.total_values / self.max_batches
                plt.plot(x, y, label=flag.name)
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()