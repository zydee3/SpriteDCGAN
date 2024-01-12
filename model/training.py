import os
import shutil
from datetime import datetime

from torch import full, zeros, FloatTensor, save
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCEWithLogitsLoss 

from model.sample import save_samples

from model.log import TrainingLog, LogFlag


DEFAULT_CONFIG_VALUES = {
    "noise_size": 100,
    "image_size": 64,
    "image_channel_size": 4,
    
    "num_epochs": 100,
    "batch_size": 32,
    "loss_function": BCEWithLogitsLoss(),
    "label_smoothing_value": 0.9,
    "gradient_clip_value": 1.0,
    "generator_train_frequency": 1,
    "discriminator_train_frequency": 1,
    "max_train_frequency": 3,

    "sample_size": 10,
    
    "out_path": None,
    "use_logging": True,
    "save_model_interval": 0,
    "save_sample_interval": 20,
    "save_loss_plot": True,
    "save_configuration": True,
    "delete_previous_saves": True,
}

TIMESTAMP_FORMAT = "%m_%d_%Y_%H_%M_%S"

class Trainer:
    def __init__(self, data, generator, discriminator, configuration=None):
        self.data = data
        self.generator = generator.cuda()
        self.discriminator = discriminator.cuda()
    
        self.configuration = self._prepare_configuration(configuration)
        self.scaler = GradScaler()
        
        self.noise_size = self.configuration["noise_size"]
        self.image_size = self.configuration["image_size"]
        self.image_channel_size = self.configuration["image_channel_size"]
        
        self.num_epochs = self.configuration["num_epochs"]
        self.batch_size = self.configuration["batch_size"]
        self.loss_function = self.configuration["loss_function"]
        self.label_smoothing_value = self.configuration["label_smoothing_value"]
        self.gradient_clip_value = self.configuration["gradient_clip_value"]
        self.generator_train_frequency = self.configuration["generator_train_frequency"]
        self.discriminator_train_frequency = self.configuration["discriminator_train_frequency"]
        self.max_train_frequency = self.configuration["max_train_frequency"]
        
        self.sample_size = self.configuration["sample_size"]
        
        self.use_logging = self.configuration["use_logging"]
        self.save_model_interval = self.configuration["save_model_interval"]
        self.save_sample_interval = self.configuration["save_sample_interval"]
        self.save_loss_plot = self.configuration["save_loss_plot"]
        self.save_configuration = self.configuration["save_configuration"]
        self.delete_previous_saves = self.configuration["delete_previous_saves"]
        
        self.out_path = self.configuration["out_path"]
        if self.out_path is not None:
            self.out_path = self._prepare_output_folder()
        
        self.fixed_noise = None
        if self.save_sample_interval > 0:
            self.fixed_noise = FloatTensor(self.batch_size, self.noise_size).uniform_(-1, 1).cuda()
        
        self.log = TrainingLog(self.num_epochs, len(self.data)) if self.use_logging else None
        

    def _prepare_output_folder(self):
        if self.delete_previous_saves and os.path.exists(self.out_path):
            shutil.rmtree(self.out_path)
        
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        
        current_timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        sub_out_path = os.path.join(self.out_path, current_timestamp)
        os.mkdir(sub_out_path)
        
        return sub_out_path
    
  
    def _prepare_configuration(self, configuration):
        prepared_configuration = None
        
        if configuration is None:
            prepared_configuration = {}
        else:
            prepared_configuration = configuration.copy()
        
        for key, value in DEFAULT_CONFIG_VALUES.items():
            prepared_configuration.setdefault(key, value)
        
        return prepared_configuration
        
    
    def _prepare_images(self, image_batch):
        image_batch_size = image_batch.size(0)
        
        real_images = image_batch.cuda()
        real_images = real_images.reshape(-1, self.image_channel_size, self.image_size, self.image_size)
    
        fake_noises = FloatTensor(image_batch_size, self.noise_size).uniform_(-1, 1).cuda()
        fake_images = self.generator(fake_noises)

        return real_images, fake_images
        

    def _prepare_labels(self, image_batch): 
        image_batch_size = image_batch.size(0)
        
        positive_labels = full((image_batch_size, 1), self.label_smoothing_value).cuda()
        negative_labels = zeros((image_batch_size, 1)).cuda()
        
        return positive_labels, negative_labels
    

    def _get_plot_save_path(self, epoch):
        if self.out_path is None:
            return None
        
        epoch_str = f"epoch_{epoch:03d}"
        out_path = f"{self.out_path}/{epoch_str}.png"
        
        return out_path
    
    
    def _train_discriminator(self, real_images, fake_images, positive_labels, negative_labels):
        total_loss = 0
                
        for i in range(self.discriminator_train_frequency):
            self.discriminator.optimizer.zero_grad()
        
            with autocast():
                eval_real = self.discriminator(real_images)
                loss_real = self.loss_function(eval_real, positive_labels)
                
                eval_fake = self.discriminator(fake_images.detach())
                loss_fake = self.loss_function(eval_fake, negative_labels)
                
                total_loss += (loss_real + loss_fake).item()
            
            loss = loss_real + loss_fake
            total_loss += loss.item()
                
            self.scaler.scale(loss).backward(retain_graph=True if i < self.discriminator_train_frequency - 1 else False)
            clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_value)
                            
        self.scaler.step(self.discriminator.optimizer)
        self.discriminator.zero_grad()
        
        return total_loss / self.discriminator_train_frequency


    def _train_generator(self, fake_images, positive_labels):
        total_loss = 0
        
        for _ in range(self.generator_train_frequency):
            self.generator.optimizer.zero_grad()
            
            with autocast():
                eval_fake = self.discriminator(fake_images)
                loss = self.loss_function(eval_fake, positive_labels)
                total_loss += loss.item()
                
            self.scaler.scale(loss).backward(retain_graph=True if _ < self.generator_train_frequency - 1 else False)
            clip_grad_norm_(self.generator.parameters(), self.gradient_clip_value)
            
        self.scaler.step(self.generator.optimizer)
        self.generator.zero_grad()
        
        return total_loss / self.generator_train_frequency

    
    def _update_progress(self, progress, epoch, generator_loss, discriminator_loss):
        epoch_str = f"Epoch [{epoch+1}/{self.num_epochs}]"
        generator_loss_str = f"Loss G: {generator_loss:.6f}"
        discriminator_loss_str = f"Loss D: {discriminator_loss:.6f}"
        
        generator_delta_loss = self.log.get_delta(LogFlag.LOG_GEN_LOSS)
        generator_delta_loss_str = f"Delta Loss G: {' ' if generator_delta_loss >= 0 else ''}{generator_delta_loss:.6f}"
        
        discriminator_delta_loss = self.log.get_delta(LogFlag.LOG_DIS_LOSS)
        discriminator_delta_loss_str = f"Delta Loss D: {' ' if discriminator_delta_loss >= 0 else ''}{discriminator_delta_loss:.6f}"
        generator_lr_str = f"LR G: {self.generator.get_learning_rate():.6f}"
        discriminator_lr_str = f"LR D: {self.discriminator.get_learning_rate():.6f}"
        
        discriminator_train_frequency_str = f"D Frequency: {self.discriminator_train_frequency}"
        generator_train_frequency_str = f"G Frequency: {self.generator_train_frequency}"
        
        tab = '\t' if epoch >= 100 else '\t\t'
        
        progress.set_description(f"{epoch_str}{tab}{discriminator_loss_str}, {generator_loss_str}, {discriminator_delta_loss_str}, {generator_delta_loss_str}, {discriminator_train_frequency_str}, {generator_train_frequency_str}, {generator_lr_str}, {discriminator_lr_str}")
    
    
    def _update_log(self, epoch, generator_loss, discriminator_loss):
        if self.log is None:
            return

        generator_lr = self.generator.get_learning_rate()
        discriminator_lr = self.discriminator.get_learning_rate()
        
        self.log.add_entry(epoch, generator_lr, discriminator_lr, generator_loss, discriminator_loss)
        
    
    def _update_training_frequency(self, discriminator_loss):
        generator_delta_loss = self.log.get_delta(LogFlag.LOG_GEN_LOSS)
        discriminator_delta_loss = self.log.get_delta(LogFlag.LOG_DIS_LOSS)
        
        # We want the generator loss to tend towards 0.0
        
        # If the generator delta loss is negative, then the generator 
        # loss is decreasing which is what we want. However, we do not
        # want it to decrease too fast so it cant overpower the 
        # discriminator.
        if generator_delta_loss <= -0.5:
            self.generator_train_frequency -= 1
        
        # If the generator delta loss is positive, then the generator
        # loss is increasing which is not what we want. We want the
        # generator loss to decrease. Therefore, we increase the
        # generator training frequency.
        if generator_delta_loss >= 0:
            self.generator_train_frequency += 1
        
        # We want the discriminator loss to stabalize around 0.5 
        
        interval_range = 0.15
        if discriminator_delta_loss < 0:
            if discriminator_loss < 0.5:
                self.discriminator_train_frequency += (0.5 - discriminator_loss) // interval_range
            else:
                self.discriminator_train_frequency -= 1
        else:
            if discriminator_loss < 0.5:
                self.discriminator_train_frequency -= 1
            else:
                self.discriminator_train_frequency += (discriminator_loss - 0.5) // interval_range
        
        self.generator_train_frequency = max(1, self.generator_train_frequency)
        self.discriminator_train_frequency = max(1, self.discriminator_train_frequency)
        
        if self.generator_train_frequency > self.discriminator_train_frequency:
            self.generator_train_frequency -= self.discriminator_train_frequency
            self.discriminator_train_frequency = 1
        elif self.generator_train_frequency < self.discriminator_train_frequency:
            self.discriminator_train_frequency -= self.generator_train_frequency
            self.generator_train_frequency = 1
            
        self.generator_train_frequency = int(min(self.generator_train_frequency, self.max_train_frequency))
        self.discriminator_train_frequency = int(min(self.discriminator_train_frequency, self.max_train_frequency))
    
    
    def _save_configuration(self):        
        if not self.save_configuration:
            return

        out_path = f"{self.out_path}/configurations.txt"
        
        with open(out_path, "w") as file:
            for key, value in self.configuration.items():
                file.write(f"{key}={value}\n")
            
        
    def _save_model(self, epoch):        
        if self.save_model_interval <= 0:
            return

        if (epoch % self.save_model_interval == 0) or (epoch == self.num_epochs - 1):
            save(self.generator.state_dict(), f"{self.out_path}/generator_{epoch+1}.pth")
            save(self.discriminator.state_dict(), f"{self.out_path}/discriminator_{epoch+1}.pth")  
    
    
    def _save_samples(self, epoch):
        if self.save_sample_interval <= 0:
            return
        
        if (epoch % self.save_sample_interval == 0) or (epoch == self.num_epochs - 1):
            save_samples(self.generator, self.fixed_noise, image_size=self.image_size, save_path=self._get_plot_save_path(epoch))

    
    def run(self):
        if self.out_path is not None:
            self._save_configuration()
            
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.num_epochs):
            progress = tqdm(enumerate(self.data), total=len(self.data))
            
            for _, (image_batch, _) in progress:                
                real_images, fake_images = self._prepare_images(image_batch)
                positive_labels, negative_labels = self._prepare_labels(image_batch)
                
                discriminator_loss = self._train_discriminator(real_images, fake_images, positive_labels, negative_labels)
                generator_loss = self._train_generator(fake_images, positive_labels)
                
                self.scaler.update()
                self._update_training_frequency(discriminator_loss)
                self._update_log(epoch, generator_loss, discriminator_loss)
                self._update_progress(progress, epoch, generator_loss, discriminator_loss)
            
            self.generator.scheduler.step()
            self.discriminator.scheduler.step()
            
            if self.out_path is not None:
                self._save_model(epoch)
                self._save_samples(epoch)
        
        if (self.out_path is not None) and (self.use_logging) and (self.save_loss_plot):
            self.log.plot(f"{self.out_path}/loss.png")
        
        return self.log
