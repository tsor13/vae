# make VAE for mnist
import torch
from torch import nn
import torch.nn.functional as F
# import MNIST
from torchvision import datasets, transforms
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
        # reparameterize
        z = self.reparameterize(mu, logvar)
        # decode
        return self.decoder(z), mu, logvar

class ConvVAE(nn.Module):
    '''
    VAE for MNIST, takes in 28x28x1 image and outputs a latent vector of size 2.
    Then, we use the latent vector to generate an image.
    '''
    def __init__(self, hidden_dim, latent_dim, activation=nn.ReLU):
        super(ConvVAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
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

    def decoder(self, z):
        z = self.decode_linear(z)
        z = z.view(-1, 64, 6, 6)
        z = self.decode_conv(z)
        # shaped as 1x30x30, need to only look at 1x28x28
        z = z[:, :, 1:-1, 1:-1]
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # encode
        z = self.encoder(x)
        mu, logvar = z[:, :self.latent_dim], z[:, self.latent_dim:]
        # reparameterize
        z = self.reparameterize(mu, logvar)
        # decode
        return self.decoder(z), mu, logvar


# dataset = 'mnist'
dataset = 'emnist'
if dataset == 'mnist':
    # load MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)
elif dataset == 'emnist':
    # load EMNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('../data', split='byclass', train=True, download=True,
                        transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('../data', split='byclass', train=False, transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)

input_dim = 28 * 28
hidden_dim = 500
latent_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, optimizer, epochs=10, c=0.01):
    model.train()
    train_loss = 0
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 28 * 28).to(device)
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
        val_loss = 0
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.view(-1, 28 * 28).to(device)
            recon_batch, mu, logvar = model(data)
            loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            # loss = F.mse_loss(recon_batch, data, reduction='mean')
            loss += c * (mu.pow(2).sum(axis=1) + logvar.exp().sum(axis=1) - logvar.sum(axis=1) - logvar.shape[1]).mean()
            val_loss += loss.item()
        print('====> Validation loss: {:.4f}'.format(
            val_loss / len(val_loader.dataset)))

def train_conv_vae(model, train_loader, val_loader, optimizer, epochs=10, c=0.01):
    model.train()
    train_loss = 0
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            # data = data.view(-1, 28 * 28).to(device)
            data = data.view(-1, 1, 28, 28).to(device)
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
        val_loss = 0
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.view(-1, 1, 28, 28).to(device)
            recon_batch, mu, logvar = model(data)
            loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            # loss = F.mse_loss(recon_batch, data, reduction='mean')
            loss += c * (mu.pow(2).sum(axis=1) + logvar.exp().sum(axis=1) - logvar.sum(axis=1) - logvar.shape[1]).mean()
            val_loss += loss.item()
        print('====> Validation loss: {:.4f}'.format(
            val_loss / len(val_loader.dataset)))



if __name__ == '__main__':
    input_dim = 28 * 28
    hidden_dim = 1024
    # hidden_dim = 128
    latent_dim = 20
    # c = 0.001
    c = 0.0001
    # c = 0
    # activation = nn.ReLU
    activation = nn.SiLU

    # model = VAE(input_dim, hidden_dim, latent_dim, activation).to(device)
    # # load params
    # # model.load_state_dict(torch.load('vae.pth'))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # train(model, train_loader, test_loader, optimizer, epochs=10, c=c)
    # # save model
    # torch.save(model.state_dict(), 'vae.pth')
    
    # train with convolutional VAE
    model = ConvVAE(hidden_dim, latent_dim, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_conv_vae(model, train_loader, test_loader, optimizer, epochs=10, c=c)
    # save model
    torch.save(model.state_dict(), 'conv_vae.pth')