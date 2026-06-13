"""

"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import clip 
from torch.cuda.amp import autocast,GradScaler
from torch.nn.utils import spectral_norm
label_to_text={
    0: "a photo of an airplane in the sky",
    1: "a photo of a car on the road",
    2: "a photo of a bird",
    3: "a photo of a cat",
    4: "a photo of a deer in nature",
    5: "a photo of a dog",
    6: "a photo of a frog",
    7: "a photo of a horse",
    8: "a photo of a ship on water",
    9: "a photo of a truck on the road"
    
    }
class Discriminator(nn.Module):
    def __init__(self, clip_dim, feature_d):
        super().__init__()
        
        self. text_pro = nn.Linear(clip_dim, 64 * 64)
        
        self.disc = nn.Sequential(
            nn.Conv2d(4, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d * 2, 4, 2, 1),
            self._block(feature_d * 2, feature_d * 4, 4, 2, 1),
            self._block(feature_d * 4, feature_d * 8, 4, 2, 1),
            self._block(feature_d * 8, 1, 4, 2, 0),
            nn.Sigmoid()            
            )
    def _block(self, in_ch, out_ch, kernel, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding)),
            nn.LeakyReLU(0.2)
            )
    def forward(self, image, text_emb):
        text_map = self.text_pro(text_emb)
        text_map = text_map.view(-1, 1, 64, 64)
        x = torch.cat([image,text_map],dim = 1)
        return self.disc(x)
class Generator(nn.Module):
    def __init__(self, z_dim, clip_dim, feature_g):
        super().__init__()
        self.first = nn.Sequential(
            nn.ConvTranspose2d(z_dim+clip_dim, feature_g * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU()
            )
        
        self.gen = nn.Sequential(
            self._block(feature_g * 8, feature_g * 4),
            self._block(feature_g * 4 , feature_g * 2),
            self._block(feature_g * 2, feature_g),
            self._block(feature_g, feature_g // 2),
            nn.Conv2d(feature_g // 2, ch, 3, 1, 1),
            nn.Tanh()
            )
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor= 2, mode= "nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
            )
        
    def forward(self, noise, text_emb):
        x = torch.cat([noise,text_emb], dim= 1)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.first(x)
        return self.gen(x)
    
    

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
z_dim = 128
clip_dim = 512

gen_input_dim = z_dim + clip_dim
ch = 3
feature_g = 64
feature_d = 64

batch_size = 64
epochs = 50

clip_model,_ = clip.load("ViT-B/32", device = device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad =  False
disc = Discriminator(clip_dim, feature_d).to(device)
gen = Generator(z_dim, clip_dim, feature_g).to(device)

fixed_noise = torch.randn((batch_size,z_dim)).to(device)

fixed_text = [label_to_text[i % 10] for i in range(batch_size)]

fixed_token = clip.tokenize(fixed_text).to(device)

with torch.no_grad():
    fixed_emb = clip_model.encode_text(fixed_token).float()
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
datasets = Datasets.CIFAR10(root="dataset/", transform=transform, download=True)

loader = DataLoader(datasets,batch_size = batch_size, shuffle = True, num_workers= 4)
optimizer_d = optim.Adam(disc.parameters(),lr = lr, betas = (0.5, 0.999))
optimizer_g = optim.Adam(gen.parameters(),lr = lr, betas = (0.5, 0.999))
criterion = nn.BCELoss()
start_epoch = 1
#uncomment the below to load the saved model and continue training 

'''
checkpoint = torch.load('checkpoint.pth')
gen.load_state_dict(checkpoint['generator'])
disc.load_state_dict(checkpoint['discriminator'])
optimizer_g.load_state_dict(checkpoint['optimizer_g'])
optimizer_d.load_state_dict(checkpoint['optimizer_d'])
start_epoch = checkpoint['epoch'] + 1
'''


fixed_real,_=next(iter(loader))
print(fixed_real.shape)
with torch.no_grad():
    all_text_embeds = {}
    for label, text in label_to_text.items():
        tokens = clip.tokenize([text]).to(device)
        all_text_embeds[label] = clip_model.encode_text(tokens).float()
from tqdm import tqdm
for epoch in range(start_epoch,start_epoch+epochs+1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{start_epoch + epochs}")
    
    for batch_idx, (real,labels) in enumerate(pbar):
        real =  real.to(device)
        batch_size = real.shape[0]
        #Text embeddings for this batch
        text_emb = torch.cat([all_text_embeds[l.item()] for l in labels])
            
        #Training the Disctiminator
        
        noise = torch.randn(batch_size,z_dim).to(device)
        fake = gen(noise, text_emb)
        disc_real = disc(real, text_emb).view(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach(), text_emb).view(-1)
        loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_d = (loss_D_real + loss_d_fake)/2
        disc.zero_grad()
        loss_d.backward()
        optimizer_d.step()
    
        #Train Generator
        
        output = disc(fake, text_emb).view(-1)
        loss_g = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        pbar.set_postfix({
            "G loss": f"{loss_g:.4f}",
            "D loss": f"{loss_d:.4f}",
           
        })
        
            
    with torch.no_grad():
        fake = gen(fixed_noise, fixed_emb)
        data = fixed_real
        img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize= True)
        img_grid_real = torchvision.utils.make_grid(data[:16], normalize= True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(img_grid_real.permute(1, 2, 0).cpu())
        axes[0].set_title(f"Real — Epoch {epoch}")
        axes[0].axis("off")
        
        axes[1].imshow(img_grid_fake.permute(1, 2, 0).cpu())
        axes[1].set_title(f"Fake — Epoch {epoch}")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()
    
            
#%%

torch.save({
    'generator': gen.state_dict(),
    'discriminator': disc.state_dict(),
    'optimizer_g': optimizer_g.state_dict(),
    'optimizer_d': optimizer_d.state_dict(),
    'epoch': epoch
}, 'checkpoint.pth')
#%%
gen.eval()
with torch.no_grad():
    fake = gen(fixed_noise, fixed_emb)
    data = fixed_real
    img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize= True)
   
    plt.figure(figsize=(12, 4))
    plt.imshow(img_grid_fake.permute(1, 2, 0).cpu())
    plt.title("Generated image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

