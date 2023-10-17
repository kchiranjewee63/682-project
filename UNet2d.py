import os
from models.UNet2d import UNet2d
from torch.optim import Adam
from utils.Meter import CrossEntropyDiceLoss
from training import Trainer
from data_processing import get_datasets

import warnings
warnings.simplefilter("ignore")

def get_BRATS_images_and_masks():
    image_names = os.listdir(os.path.join('BRATS_DATA', 'trainA'))
    
    images = [(os.path.join('BRATS_DATA', 'trainA', image_name), os.path.join('BRATS_DATA', 'trainB', image_name.replace("t1n", "t2f"))) for image_name in image_names]
    
    masks = [os.path.join('BRATS_DATA', 'maskA', image_name) for image_name in image_names]
    
    return images, masks

train_dataloader, val_dataloader = get_datasets(get_BRATS_images_and_masks, None, batch_size = 32, center_crop_size = (224, 224))

model = {"model": UNet2d(in_channels = 2, n_classes = 3, n_channels = 8).to('cuda'), "name":f"UNet2d"}

trainer = Trainer(net = model["model"],
                  net_name = model["name"],
                  criterion = CrossEntropyDiceLoss(),
                  lr = 1e-3,
                  num_epochs = 500,
                  optimizer = Adam,
                  load_prev_model = True,
                  train_dataloader = train_dataloader,
                  val_dataloader = val_dataloader)
    
trainer.run()