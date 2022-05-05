import torch
import torch.distributed as dist
import torch.nn as nn
from syncbn import SyncBatchNorm
import torch.testing
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f'device is {device}')


def test_batch_norm(input):
    input = input.to(device)
    torch_batch_norm = nn.BatchNorm1d(128, affine=False).to(device)(input)
    my_batch_norm = SyncBatchNorm(128).to(device)(input)

    logging.info(
        f'Rank{dist.get_rank()} mean is {torch.isclose(torch_batch_norm, my_batch_norm, rtol=1e-06, equal_nan=True).float().mean()}')

    torch.testing.assert_close(torch_batch_norm, my_batch_norm)


def init_process(local_rank, fn, backend):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    fn()


def main_test_function():
    input = torch.load('input0_batch_64.pt')
    test_batch_norm(input)


# don't find any way how to use pytest with torchrun, so decided run simple python script
if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=main_test_function, backend="gloo")
