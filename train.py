import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader, annotations
from models import Generator, Discriminator

# Hyperparameters
latent_dim = 100
num_classes = 3  # Assuming 3 different locations
img_shape = (3, 64, 64)  # Assuming RGB images of size 64x64
batch_size = 2
num_epochs = 200

# Load Data
video_folder = 'path/to/videos'
dataloader = get_dataloader(video_folder, annotations, batch_size)

# Initialize models
generator = Generator(latent_dim, num_classes, img_shape)
discriminator = Discriminator(num_classes, img_shape)

# Loss function
adversarial_loss = nn.MSELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training
for epoch in range(num_epochs):
    for i, (frames, locations, counts) in enumerate(dataloader):
        batch_size = frames.size(0)
        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        real_imgs = frames[:, 0, :, :, :]  # Using the first frame for simplicity

        # Generate noise and labels
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels, counts)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs, locations, counts), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels, counts), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels, counts), valid)

        g_loss.backward()
        optimizer_G.step()

        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
