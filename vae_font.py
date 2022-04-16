# make VAE for mnist
import torch
from torch import nn
import torch.nn.functional as F
from fontdataset import FontDataset
from pdb import set_trace as breakpoint

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation=nn.ReLU):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode
        z = self.encoder(x)
        mu, logvar = z[:, :self.latent_dim], z[:, self.latent_dim:]
        # clip logvar to avoid exploding gradients
        logvar = torch.clamp(logvar, min=-15, max=15)
        # reparameterize
        z = self.reparameterize(mu, logvar)
        # decode
        return self.decoder(z), mu, logvar

class ConvVAE(nn.Module):
    '''
    VAE for MNIST, takes in 28x28x1 image and outputs a latent vector of size 2.
    Then, we use the latent vector to generate an image.
    '''
    def __init__(self, size, hidden_dim, latent_dim, activation=nn.ReLU):
        super(ConvVAE, self).__init__()
        self.size = size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        if size == 28:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2),
                activation(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                activation(),
                # flatten
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, latent_dim * 2),
            )

            self.decode_linear = nn.Sequential(
                nn.Linear(latent_dim, 64 * 6 * 6),
                activation(),
            )
            self.decode_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
                nn.Sigmoid()
            )
        elif size == 256:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2),
                activation(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                activation(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                activation(),
                nn.Conv2d(128, 128, kernel_size=4, stride=2),
                activation(),
                # flatten
                nn.Flatten(),
                nn.Linear(128 * 14 * 14, latent_dim * 2),
            )

            self.decode_linear = nn.Sequential(
                nn.Linear(latent_dim, 128 * 14 * 14),
                activation(),
            )
            self.decode_conv = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
                nn.Sigmoid()
            )

    def decoder(self, z):
        z = self.decode_linear(z)
        if self.size == 28:
            z = z.view(-1, 64, 6, 6)
            z = self.decode_conv(z)
            # shaped as 1x30x30, need to only look at 1x28x28
            z = z[:, :, 1:-1, 1:-1]
        elif self.size == 256:
            z = z.view(-1, 128, 14, 14)
            z = self.decode_conv(z)
            new_z = torch.zeros(z.shape[0], 1, 256, 256).to(z.device)
            new_z[:, :, 1:-1, 1:-1] = z
            z = new_z
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # encode
        z = self.encoder(x)
        mu, logvar = z[:, :self.latent_dim], z[:, self.latent_dim:]
        # clip logvar to avoid exploding gradients
        logvar = torch.clamp(logvar, min=-15, max=15)
        # reparameterize
        z = self.reparameterize(mu, logvar)
        # decode
        return self.decoder(z), mu, logvar


# dataset = 'mnist'
# dataset = 'emnist'
# size = 28
# batch_size=64
size = 256
batch_size=16

train_dataset = FontDataset(f'font_images_{size}')
test_dataset = FontDataset(f'font_images_{size}')
# make loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

input_dim = size**2
hidden_dim = 500
latent_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, test_loader, optimizer, epochs=10, c=0.01):
    model.train()
    train_loss = 0
    for epoch in range(epochs):
        for batch_idx, (data, _, _) in enumerate(train_loader):
            data = data.view(-1, size**2).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            # loss = F.mse_loss(recon_batch, data, reduction='mean')
            loss += c * (mu.pow(2).sum(axis=1) + logvar.exp().sum(axis=1) - logvar.sum(axis=1) - logvar.shape[1]).mean()
            # loss += c * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).to(device)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        # validate
        # test_loss = 0
        # for batch_idx, (data, _, _) in enumerate(test_loader):
        #     data = data.view(-1, size**2).to(device)
        #     recon_batch, mu, logvar = model(data)
        #     loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
        #     # loss = F.mse_loss(recon_batch, data, reduction='mean')
        #     loss += c * (mu.pow(2).sum(axis=1) + logvar.exp().sum(axis=1) - logvar.sum(axis=1) - logvar.shape[1]).mean()
        #     test_loss += loss.item()
        # print('====> Validation loss: {:.4f}'.format(
        #     test_loss / len(test_loader.dataset)))

def train_conv_vae(model, train_loader, test_loader, optimizer, epochs=10, c=0.01):
    model.train()
    train_loss = 0
    for epoch in range(epochs):
        for batch_idx, (data, _, _) in enumerate(train_loader):
            # data = data.view(-1, 28 * 28).to(device)
            data = data.view(-1, 1, size, size).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            # loss = F.mse_loss(recon_batch, data, reduction='mean')
            loss += c * (mu.pow(2).sum(axis=1) + logvar.exp().sum(axis=1) - logvar.sum(axis=1) - logvar.shape[1]).mean()
            # loss += c * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).to(device)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        # validate
        # test_loss = 0
        # for batch_idx, (data, _, _) in enumerate(test_loader):
        #     data = data.view(-1, 1, size, size).to(device)
        #     recon_batch, mu, logvar = model(data)
        #     loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
        #     # loss = F.mse_loss(recon_batch, data, reduction='mean')
        #     loss += c * (mu.pow(2).sum(axis=1) + logvar.exp().sum(axis=1) - logvar.sum(axis=1) - logvar.shape[1]).mean()
        #     test_loss += loss.item()
        # print('====> Validation loss: {:.4f}'.format(
        #     test_loss / len(test_loader.dataset)))



if __name__ == '__main__':
    input_dim = size**2
    hidden_dim = 1024
    # hidden_dim = 128
    latent_dim = 128
    # c = 0.001
    c = 0.0001
    # c = 0.00001
    # c = 0
    # activation = nn.ReLU
    activation = nn.SiLU
    # activation = nn.Sigmoid

    # model = VAE(input_dim, hidden_dim, latent_dim, activation).to(device)
    # # load params
    # # model.load_state_dict(torch.load('vae.pth'))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # train(model, train_loader, test_loader, optimizer, epochs=1, c=c)
    # # save model
    # torch.save(model.state_dict(), 'vae.pth')
    
    # # train with convolutional VAE
    model = ConvVAE(size, hidden_dim, latent_dim, activation).to(device)
    model.load_state_dict(torch.load('conv_vae.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_conv_vae(model, train_loader, test_loader, optimizer, epochs=5, c=c)
    # save model
    torch.save(model.state_dict(), 'conv_vae.pth')