import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import sys
import random
import logging
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s – %(levelname)s: \t %(message)s')


def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    def __init__(self, n_process: int, device):
        assert 32 % n_process == 0
        super().__init__()
        self.device = device

        self.conv1 = nn.Conv2d(3, 32 // n_process, 3, 1)
        self.conv2 = nn.Conv2d(32, 32 // n_process, 3, 1)
        self.fc1 = nn.Linear(6272, 128 // n_process)
        self.fc2 = nn.Linear(128, 100 // n_process)

        self.bn1 = nn.BatchNorm1d(num_features=128 // n_process)

    def forward(self, x):
        x = self.conv1(x)

        x_gathered = [torch.zeros(x.size()).to(self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(x_gathered, x)
        x_gathered[dist.get_rank()] = x  # here happens fucking magic, now autograd is working
        x = torch.cat(x_gathered, dim=1)

        x = F.relu(x)

        x = self.conv2(x)

        x_gathered = [torch.zeros(x.size()).to(self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(x_gathered, x)
        x_gathered[dist.get_rank()] = x  # here happens fucking magic, now autograd is working
        x = torch.cat(x_gathered, dim=1)

        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        logging.debug(f'before fc1{x.size()}')
        x = self.fc1(x)

        x = F.relu(x)
        x = self.bn1(x)

        logging.debug(f'after fc1 + batchnorm {x.size()}')

        x_gathered = [torch.zeros(x.size()).to(self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(x_gathered, x)
        x_gathered[dist.get_rank()] = x  # here happens fucking magic, now autograd is working
        x = torch.cat(x_gathered, dim=1)
        logging.debug(f'after fc1 cat{x.size()}')

        x = self.fc2(x)
        x_gathered = [torch.zeros(x.size()).to(self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(x_gathered, x)
        x_gathered[dist.get_rank()] = x
        x = torch.cat(x_gathered, dim=1)

        return x


def run_training(rank, size):
    torch.manual_seed(rank)
    random.seed(rank)

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=f'part03_metrics/{datetime.datetime.now().strftime("%H:%M:%S")}')

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataset = CIFAR100(
        "./data_train",
        transform=transform_cifar,
        train=True,
        download=True,
    )
    train_loader = DataLoader(dataset, batch_size=64)
    device = torch.device('cuda', rank)
    ## device = torch.device('cuda', 0)  # to debug in `gloo`

    model = Net(n_process=dist.get_world_size(), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    model.to(device)
    model.train()
    for epoch in range(10):

        epoch_loss = 0
        start_time = datetime.datetime.now()
        for batch_num, (data, target) in enumerate(tqdm(train_loader, disable=True)):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            optimizer.step()

            train_acc = (output.argmax(dim=1) == target).float().mean()

        if dist.get_rank() == 0:
            logging.info(f"epoch {epoch} time {round((datetime.datetime.now() - start_time).total_seconds(), 3)} \t "
                         f"loss is {loss}, acc is: {train_acc}")

            writer.add_scalar(f'WorldSize{dist.get_world_size()}/Train loss',
                              scalar_value=epoch_loss / len(train_loader),
                              global_step=epoch)
            writer.add_scalar(f'WorldSize{dist.get_world_size()}/Train accuracy1', scalar_value=train_acc,
                              global_step=epoch)

        logging.debug(epoch_loss)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    # init_process(local_rank, fn=run_training, backend='gloo')
    init_process(local_rank, fn=run_training, backend='nccl')
