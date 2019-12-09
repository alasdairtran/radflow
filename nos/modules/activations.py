import math

import torch


def gelu(x):
    """Apply the GELU activation function.

    This GELU is used in GPT and XLNet. See https://arxiv.org/abs/1606.08415.
    Replace it with the native Pytorch function when Pytorch 1.2 is out.
    """
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi)
                                  * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


def gelu_2(x):
    """Apply of the GELU activation function.

    This GELU is used in BERT. See https://arxiv.org/abs/1606.08415. Replace it
    with the native Pytorch function when Pytorch 1.2 is out.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    """Apply the Swish activation function.

    Swish outperforms ReLU and GELU. See https://arxiv.org/pdf/1710.05941.pdf.
    """
    return x * torch.sigmoid(x)
