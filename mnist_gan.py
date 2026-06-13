"""
The purpose of this code:

    It is to get the idea about how to build a generator and discriminator from scratch and train it.

What it does:
    
    The generator generates a image from a random noise, when you are starting to train you will see the plot showing the random noises in the image.

How it works:
    
    The discriminator sees both the real image and fake image and it produces loss for both real and fake image and the average of them will be the discriminator loss , the generator gets output from discriminator for the fake image it generated and calculates its own loss.

Result:

    You will see the digits starting to have the shape and less noise after 20 epochs. But it will not match the real digits, why? the generator learns to generate digits without much noise, but it still does not know which is the right image for right digit.

What is next:

    Thats why I am building another GAN which is going to be for CIFAR-10 dataset images ,with colours and I am going to add description for each class in the dataset and train it with text to get the better output.
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
 
class Discriminator(nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(feat, 512),
            #nn.BatchNorm1d(512), #tried to check if adding batchnormalization works well, but it added more noise to the edges , you can try to uncomment and see the result.
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()            
            )
    def forward(self, x):
        return self.disc(x)
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, img_dim),
            nn.Tanh(),
            )
    def forward(self, x):
        return self.gen(x)
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
z_dim = 128
img_dim = 28 * 28 * 1
batch_size = 32
epochs = 10
disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
start_epoch = 1
#Uncomment for continue training
'''
checkpoint = torch.load('Model_weight/mnist_gan_weights.pth')
gen.load_state_dict(checkpoint['generator'])
disc.load_state_dict(checkpoint['discriminator'])
optimizer_g.load_state_dict(checkpoint['optimizer_g'])
optimizer_d.load_state_dict(checkpoint['optimizer_d'])
start_epoch = checkpoint['epoch'] + 1
'''



fixed_noise = torch.randn((batch_size,z_dim)).to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ])
datasets = Datasets.MNIST(root= "dataset/", transform=transform, download = " True")
loader =  DataLoader(datasets,batch_size = batch_size, shuffle = True)
optimizer_d = optim.Adam(disc.parameters(),lr = 3e-5)
optimizer_g = optim.Adam(gen.parameters(),lr = lr)
criterion = nn.BCELoss()
fixed_real,_ = next(iter(loader))
from tqdm import tqdm

for epoch in range(start_epoch,start_epoch + epochs + 1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs+90}")
    
    for batch_idx, (real,_) in enumerate(pbar):
        real =  real.view(-1,784).to(device)
        batch_size = real.shape[0]
        
        #Training the Disctiminator
        
        noise = torch.randn(batch_size,z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_d = (loss_D_real + loss_d_fake)/2
        disc.zero_grad()
        loss_d.backward(retain_graph = True)
        optimizer_d.step()
    
        #Train Generator
        
        output = disc(fake).view(-1)
        loss_g = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        pbar.set_postfix({
            "G loss": f"{loss_g:.4f}",
            "D loss": f"{loss_d:.4f}",
           
        })
        
            
    with torch.no_grad():
        fake = gen(fixed_noise).reshape(-1,1,28,28)
        data = fixed_real.reshape(-1,1,28,28)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize= True)
        img_grid_real = torchvision.utils.make_grid(data, normalize= True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(img_grid_real.permute(1, 2, 0).cpu())
        axes[0].set_title(f"Real — Epoch {epoch}")
        axes[0].axis("off")
        
        axes[1].imshow(img_grid_fake.permute(1, 2, 0).cpu())
        axes[1].set_title(f"Fake — Epoch {epoch}")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()
    

torch.save({
    'generator': gen.state_dict(),
    'discriminator': disc.state_dict(),
    'optimizer_g': optimizer_g.state_dict(),
    'optimizer_d': optimizer_d.state_dict(),
    'epoch': epoch
}, 'Model_weights/mnist_gan_weights.pth')       

#The below code is to Generate image directly without discriminator involvement
gen.eval()
noise = torch.randn(32, z_dim).to(device)
with torch.no_grad():
    fake = gen(noise).reshape(-1,1,28,28)
    img_grid_fake = torchvision.utils.make_grid(fake, normalize= True)
    plt.imshow(img_grid_real.permute(1, 2, 0).cpu())
    plt.title("Generated Image from noise")
    plt.axis("off")
    plt.tight_layout()
    plt.show()