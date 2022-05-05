import logging
import os
import time
from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
from syncbn import SyncBatchNorm
from torch.utils.tensorboard import SummaryWriter
import sys

torch.set_num_threads(1)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s – %(levelname)s: \t %(message)s')


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)

    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        # self.bn1 = nn.BatchNorm1d(128, affine=False)  # to be replaced with SyncBatchNorm
        self.bn1 = SyncBatchNorm(num_features=128)
        # self.bn1 = nn.SyncBatchNorm(128, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # torch.save(x, f'input0_batch_64.pt')
        # exit(0)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def validate(val_loader, model, device, *args):
    logging.debug(f'{dist.get_rank()} in validate')
    model.eval()

    with torch.no_grad():
        val_accuracy_top1, val_accuracy_top5 = 0., 0.
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            batch_pred = model(data)

            val_accuracy_top1 += (batch_pred.argmax(dim=1) == target).float().mean()

            _, top5 = batch_pred.topk(5, 1, True, True)
            target_expanded = target.view(-1, 1).expand_as(top5)

            val_accuracy_top5 += (target_expanded == top5).float().sum() / len(data)
            logging.debug(f'accuracy {val_accuracy_top1} and {val_accuracy_top5}')

        val_accuracy_top1 /= len(val_loader)
        val_accuracy_top5 /= len(val_loader)

    return val_accuracy_top1, val_accuracy_top5


tensorboard_time = datetime.now().strftime("%H:%M:%S")
# experiment_name_prefix = f'torch_syncbatchnorm_EXP'  # замерка торчвого sync BatchNorm
experiment_name_prefix = f'MyBatchVersion'  # для замерки моего батчнорма


def run_training(rank, size, is_nccl=True,
                 tag_for_tensorboard=lambda: f'Rank{dist.get_rank()}_{experiment_name_prefix}'):
    torch.manual_seed(0)
    writer = SummaryWriter(
        log_dir=f'cifar100_metrics/{experiment_name_prefix}_{dist.get_world_size()}_{tensorboard_time}')

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

    train_loader = DataLoader(dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=64)

    if dist.get_rank() == 0:
        test_loader = DataLoader(CIFAR100('./data_test', train=False, transform=transform_cifar, download=True),
                                 shuffle=False, batch_size=64)

        logging.info(f'train_loader len is {len(train_loader)} \t test_loader len is {len(test_loader)}')

    model = Net()
    device = torch.device("cuda", dist.get_rank() if is_nccl else 0) if torch.cuda.is_available() else torch.device(
        "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(train_loader)

    for epoch in tqdm(range(50), disable=dist.get_rank() != 0):
        epoch_loss = torch.zeros((1,), device=device)
        train_acc = None
        start_time = datetime.now()

        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            average_gradients(model)
            optimizer.step()

            train_acc = (output.argmax(dim=1) == target).float().mean()

            if dist.get_rank() == 0:
                writer.add_scalar(f'{tag_for_tensorboard()}/Train loss', scalar_value=epoch_loss / num_batches,
                                  global_step=epoch)
                writer.add_scalar(f'{tag_for_tensorboard()}/Train accuracy1', scalar_value=train_acc, global_step=epoch)

        if dist.get_rank() == 0:
            val_accuracy1, val_accuracy5 = validate(test_loader, model, device)

            writer.add_scalar(f'{tag_for_tensorboard()}/val_accuracy1', scalar_value=val_accuracy1,
                              global_step=epoch)
            writer.add_scalar(f'{tag_for_tensorboard()}/val_accuracy5', scalar_value=val_accuracy5,
                              global_step=epoch)

            writer.add_scalar(f'Time', scalar_value=(round((datetime.now() - start_time).total_seconds(), 3)),
                              global_step=epoch)

        logging.debug(f'{dist.get_rank()} in barrier')
        dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    # init_process(local_rank, fn=run_training, backend="gloo")
    init_process(local_rank, fn=run_training, backend="nccl")  # replace with "nccl" when testing on GPUs
