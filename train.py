import torch
from deep_trainer import PytorchTrainer
from torch.utils.data import DataLoader
from dataset import Mydata
from torchvision import transforms
from loss import SBloss
from unet_model import UNet,init_weights
from torch import optim
from normpi import normpi


# norm_mean = [0.5002]
# norm_std = [0.2578]
train_transform = transforms.Compose([
    transforms.ToTensor(),
    normpi()
    # transforms.Normalize(norm_mean, norm_std),
])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    normpi()
])

train_dataset = Mydata('../../autodl-tmp/imgs/train', '../../autodl-tmp/labels/train', transform=train_transform)
valid_dataset = Mydata('../../autodl-tmp/imgs/val', '../../autodl-tmp/labels/val', transform=valid_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=6, shuffle=True)

net = UNet(n_channels=1, n_classes=5,bilinear=True)
checkpoint = torch.load('./exp/checkpoints/best.ckpt')
net.load_state_dict(checkpoint['model'])
# net.apply(init_weights)

device = torch.device('cuda:0')
net.to(device)
loss = SBloss(1.1, 1.1, 1, 1.5, 1.5)

optimizer = optim.AdamW(net.parameters(), lr=0.001)
# optimizer.load_state_dict(checkpoint["optimizer"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0,
                                                 last_epoch=-1)  
# scheduler.load_state_dict(checkpoint["scheduler"])
trainer = PytorchTrainer(net, optimizer, scheduler, save_mode="small", device=device,use_amp=True)
trainer.train(64, train_loader, loss, val_loader=valid_loader)
