import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import numpy as np
import torchvision
from network import Generator,Discriminator
from config import Config


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
generator = Generator(Config.img_shape,Config.z_dim).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(),
                                       lr=Config.LR,
                                       betas=(0.5, 0.999))

discriminator = Discriminator(Config.img_shape).to(device)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                           lr=Config.LR,
                                           betas=(0.5,0.999))

d_loss = nn.BCELoss()
g_loss = nn.BCELoss()

path = os.path.join(os.getcwd(),"../files")

# data processing
# MNIST Dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = torchvision.datasets.MNIST(root=path,
                                           train=True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=path,
                                          train=False,
                                          transform=transform,
                                          download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=Config.BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=Config.BATCH_SIZE,
                                          shuffle=False,
                                          drop_last=True)


def visualize_images(images, image_grid_rows=4, image_grid_columns=4):
    # gen_imgs = images*0.5 +0.5
    """
    fig,axs = plt.subplots(image_grid_rows,
                           image_grid_columns,
                           figsize=(4,4),
                           sharey=True,
                           sharex=True)
    """
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            plt.subplot(image_grid_rows,image_grid_columns,i*image_grid_columns+j+1)
            plt.imshow(images[cnt,0,:,:], cmap='gray')
            # axs[i,j].imshow(gen_imgs[cnt,0,:,:].detach(), cmap='gray')
            # axs[i,j].axis('off')
            cnt += 1
    plt.show()


def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    # import ipdb;ipdb.set_trace()
    # z = ((torch.rand(image_grid_rows*image_grid_columns, Config.z_dim)-0.5)/0.5).to(device)
    z = torch.normal(0,1,(image_grid_rows*image_grid_columns, Config.z_dim)).to(device)
    # z = torch.randn(image_grid_rows*image_grid_columns,Config.z_dim).to(device)
    gen_imgs = generator(z).cpu()
    gen_imgs = gen_imgs.view(-1,Config.channels,Config.img_rows,Config.img_cols)
    # gen_imgs = gen_imgs*0.5 +0.5
    """
    fig,axs = plt.subplots(image_grid_rows,
                           image_grid_columns,
                           figsize=(4,4),
                           sharey=True,
                           sharex=True)
    """
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            plt.subplot(image_grid_rows,image_grid_columns,i*image_grid_columns+j+1)
            plt.imshow(gen_imgs[cnt,0,:,:].detach(), cmap='gray')
            # axs[i,j].imshow(gen_imgs[cnt,0,:,:].detach(), cmap='gray')
            # axs[i,j].axis('off')
            import ipdb;ipdb.set_trace()
            cnt += 1
    plt.show()

def train(epochs, batch_size):

    g_loss_history = []
    d_loss_history = []
    epoch = 0

    while epoch < Config.iterations:
        for idx,images in enumerate(train_loader):
            """
            train discriminator
            """
            # import ipdb;ipdb.set_trace()
            # visualize_images(images)
            images = images[0].view(images[0].shape[0],-1)
            images = images.to(device)
            # import ipdb;ipdb.set_trace()
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            # generate random noise samples
            # normal distribution
            # noise = torch.randn(Config.BATCH_SIZE,Config.z_dim).to(device)
            noise = torch.normal(0,1,(Config.BATCH_SIZE,Config.z_dim)).to(device)
            # uniform distribution
            # noise = ((torch.rand(Config.BATCH_SIZE,Config.z_dim)-0.5)/0.5).to(device)

            d_fake = generator(noise)
            # gen_imgs = gen_imgs.view(Config.BATCH_SIZE,-1)
            d_fake_score = discriminator(d_fake)
            d_real_score = discriminator(images)

            label_fake = torch.zeros((Config.BATCH_SIZE,1)).to(device)
            label_real = torch.ones((Config.BATCH_SIZE,1)).to(device)

            loss_discriminator = d_loss(d_fake_score,label_fake) + \
                    d_loss(d_real_score,label_real)

            loss_discriminator.backward()
            discriminator_optimizer.step()

            """
            train generator
            """
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            # noise = torch.randn(Config.BATCH_SIZE,Config.z_dim).to(device)
            # uniform distribution
            noise = torch.normal(0,1,(Config.BATCH_SIZE,Config.z_dim)).to(device)
            # noise = ((torch.rand(Config.BATCH_SIZE,Config.z_dim)-0.5)/0.5).to(device)
            g_fake = generator(noise)
            g_fake_score = discriminator(g_fake)

            loss = g_loss(g_fake_score,label_real)
            loss.backward()
            generator_optimizer.step()

            epoch += 1
            g_loss_history.append(loss_discriminator.item())
            d_loss_history.append(loss.item())

            if epoch % Config.SAMPLE_INTERVAL == 0:
                # g_loss_history.append(loss_discriminator.item())
                # d_loss_history.append(loss.item())
                print(f'current epoch: {epoch}, \
                        index: {idx}, \
                        g_loss: {loss.item()}, \
                        d_loss: {loss_discriminator.item()}')
                # sample_images(generator)
                # plt.plot(g_loss_history,c='b')
                # plt.plot(d_loss_history,c='r')
                # plt.show()
    plt.plot(g_loss_history,c='b')
    plt.plot(d_loss_history,c='r')
    plt.show()
    sample_images(generator)

if __name__ == "__main__":
    train(Config.EPOCHS, Config.BATCH_SIZE)
