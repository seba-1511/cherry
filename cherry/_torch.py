#!/usr/bin/env python3

import torch as th
import numpy as np

from cherry._utils import EPS


def totensor(array, dtype=None):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/_torch.py)

    **Description**

    Converts the argument `array` to a torch.tensor 1xN, regardless of its
    type or dimension.

    **Arguments**

    * **array** (int, float, ndarray, tensor) - Data to be converted to array.
    * **dtype** (dtype, *optional*, default=None) - Data type to use for representation.
    By default, uses `torch.get_default_dtype()`.

    **Returns**

    * Tensor of shape 1xN with the appropriate data type.

    **Example**

    ~~~python
    array = [5, 6, 7.0]
    tensor = cherry.totensor(array, dtype=th.float32)
    array = np.array(array, dtype=np.float64)
    tensor = cherry.totensor(array, dtype=th.float16)
    ~~~

    """
    if dtype is None:
        dtype = th.get_default_dtype()
    if isinstance(array, (list, tuple)):
        array = th.cat([totensor(x) for x in array], dim=0)
    else:
        if isinstance(array, int):
            array = float(array)
        if isinstance(array, float):
            array = [array, ]
        if isinstance(array, list):
            array = np.array(array)
        if isinstance(array, (np.ndarray,
                              np.bool_,
                              np.float32,
                              np.float64,
                              np.int32,
                              np.int64)):
            if array.dtype == np.bool_:
                array = array.astype(np.uint8)
        if not isinstance(array, th.Tensor):
            array = th.tensor(array, dtype=dtype)
    while array.ndim < 2:
        array = array.unsqueeze(0)
    return array


def normalize(tensor, epsilon=EPS):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/_torch.py)

    **Description**

    Normalizes a tensor to have zero mean and unit standard deviation values.

    **Arguments**

    * **tensor** (tensor) - The tensor to normalize.
    * **epsilon** (float, *optional*, default=1e-8) - Numerical stability constant for
    normalization.

    **Returns**

    * A new tensor, containing the normalized values.

    **Example**

    ~~~python
    tensor = torch.arange(23) / 255.0
    tensor = cherry.normalize(tensor, epsilon=1e-3)
    ~~~

    """
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + epsilon)


def onehot(x, dim):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/_torch.py)

    **Description**

    Creates a new onehot tensor of the specified dimension.

    **Arguments**

    * **x** (int, ndarray, tensor) - Index or N-dimensional tensor of indices to be one-hot encoded.
    * **dim** (int) - Size of the one-hot vector.

    **Returns**

    * A new Nxdim tensor containing one(s) at position(s) `x`, zeros everywhere else.

    **Example**

    ~~~python
    action = 2
    action = cherry.onehot(action, dim=5)

    actions = torch.tensor([[2], [1], [2]]).long()  # 3x1 tensor
    actions = cherry.onehot(actions, dim=5)  # 3x5 tensor
    ~~~

    """
    size = 1
    if isinstance(x, np.ndarray):
        size = x.shape[0]
        x = th.from_numpy(x).long()
    if isinstance(x, (int, float, np.integer, np.float)):
        x = [int(x), ]
    if isinstance(x, list):
        x = th.tensor(x).view(-1, 1).long()
    if isinstance(x, th.Tensor):
        size = x.size(0)
        x = x.long()
    onehot = th.zeros(size, dim, device=x.device)
    onehot.scatter_(1, x.view(-1, 1), 1.0)
    return onehot
