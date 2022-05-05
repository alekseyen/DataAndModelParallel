import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`

        input = input.contiguous()
        size = input.numel() / input.size(1)

        device = input.deviceда
        count = torch.tensor([size])
        mean, invstd = torch.batch_norm_stats(input, eps)

        count_all = [torch.zeros(count.size()).to(device) for _ in range(dist.get_world_size())]
        mean_all = [torch.zeros(mean.size()).to(device) for _ in range(dist.get_world_size())]
        invst_all = [torch.zeros(invstd.size()).to(device) for _ in range(dist.get_world_size())]

        logger.debug(f'{dist.get_rank()} before all_gather')

        dist.all_gather(mean_all, mean.to(device))
        dist.all_gather(invst_all, invstd.to(device))
        dist.all_gather(count_all, count.to(device))

        logger.debug(f'{dist.get_rank()} after all_gather')

        # dist.barrier()

        count_all = torch.stack(count_all).view(-1)
        mean_all = torch.stack(mean_all)
        invst_all = torch.stack(invst_all)

        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invst_all,
            running_mean,
            running_std,
            momentum,
            eps,
            count_all
        )

        ctx.save_for_backward(input, mean, invstd, count_all)
        return torch.batch_norm_elemt(input, weight=None, bias=None, mean=mean, invstd=invstd, eps=eps)

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!

        grad_output = grad_output.contiguous()
        saved_input, mean, invstd, count_all = ctx.saved_tensors
        count_all = count_all.to(dtype=torch.int, device=grad_output.device)

        sum_dy, sum_dy_xmu, _, _ = torch.batch_norm_backward_reduce(
            grad_out=grad_output,
            input=saved_input,
            mean=mean,
            invstd=invstd,
            weight=None,
            input_g=False,
            weight_g=False,
            bias_g=False
        )

        grad_input = torch.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd=invstd,
            weight=None,
            mean_dy=sum_dy,
            mean_dy_xmu=sum_dy_xmu,
            count=count_all
        )

        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def _run_bn(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.bias,
            training=self.training or not self.track_running_stats, momentum=self.momentum, eps=self.eps)

    def _sync_bn(self, input):
        if dist.get_world_size() == 1:
            logger.debug('RUNNING SIMPLE TORCH batchnorm')
            return self._run_bn(input)
        else:
            return sync_batch_norm.apply(input, self.running_mean, self.running_var, self.eps, self.momentum)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_running_stats:
            logger.debug(f'rank {dist.get_rank()} choose 1')
            self.num_batches_tracked = self.num_batches_tracked + 1

        if not self.training and self.track_running_stats:
            logger.debug(f'rank {dist.get_rank()} choose 1.1')
            return self._run_bn(input)
        else:
            logger.debug(f'rank {dist.get_rank()} choose 2')
            return self._sync_bn(input)
