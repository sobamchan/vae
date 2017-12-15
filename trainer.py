import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VAE


class Trainer(object):

    def __init__(self, args):
        self.args = args

        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor()),
                batch_size=self.args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=False,
                               transform=transforms.ToTensor()),
                batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = VAE()
        if self.args.cuda:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.args.batch_size * 784
        return BCE + KLD

    def train_one_epoch(self, epoch):
        train_loader = self.train_loader
        args = self.args

        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            self.optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0] / len(data)))
        print('=====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    def test(self, epoch):
        test_loader = self.test_loader
        args = self.args

        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss_function(recon_batch, data,
                                            mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                       recon_batch.view(args.batch_size,
                                                        1, 28, 28)[:n]])
                fname = 'results/reconstruction_' + str(epoch) + '.png'
                save_image(comparison.data.cpu(),
                           fname, nrow=n)

        test_loss /= len(test_loader.dataset)
        print('=====> Test set loss: {:.4f}'.format(test_loss))

    def train(self):
        args = self.args
        for epoch in range(1, args.epochs + 1):
            self.train_one_epoch(epoch)
            self.test(epoch)
            sample = Variable(torch.randn(64, 20))
            if args.cuda:
                sample = sample.cuda()
            sample = self.model.decode(sample).cpu()
            save_image(sample.data.view(64, 1, 28, 28),
                       './results/sample_' + str(epoch) + '.png')
