"""
@Time：2023/9/19
@Auth：Archer
@Email：hanwenyongava@163.com
@Project：PatchTST_supervised
@Content：mutil_tools
@Function：
"""
import torch
from sklearn.decomposition import PCA
from numpy import array

# shuffle data in dim=1
def rand_shuffle(tensor, dim):
    """
    在指定维度上对张量的数据进行打乱。

    参数：
    tensor (Tensor): 要打乱的张量。
    dim (int): 要打乱的维度。

    返回：
    Tensor: 包含在指定维度上打乱顺序的张量。
    """
    # 获取目标维度的大小
    dim_size = tensor.size(dim)

    # 生成随机排列的索引
    permuted_indices = torch.randperm(dim_size, device=tensor.device)

    # 使用随机排列的索引对数据进行重新排列
    shuffled_tensor = tensor.index_select(dim, permuted_indices)

    return shuffled_tensor


def order_shuffle(tensor, size=2, dim=1):
    # 将维度为1上的特征向量重新排序
    dim_sizes = list(tensor.size())
    batch_size = dim_sizes[0]
    target_size = dim_sizes[dim]
    half_n = target_size // size

    part1 = tensor[:, half_n:, :]
    part2 = tensor[:, :half_n, :]

    tensor_reordered = torch.cat([part1, part2], dim=dim)

    # size = tensor.shape[dim]
    # new_order = [i for i in range(1, size)] + [0]  # 新的顺序：2, 3, ..., 20, 21, 1
    # tensor_reordered = tensor[:, new_order, :, :]
    return tensor_reordered


def flatten(sample):
    u = torch.reshape(sample, (sample.shape[0] * sample.shape[1], sample.shape[2], sample.shape[3]))
    return u


def enhance(sample, stack_size, dim=1):
    samples = list()
    samples.append(flatten(sample))
    n_sample = sample

    if stack_size > 1:
        for i in range(stack_size-1):
            n_sample = order_shuffle(n_sample,size=stack_size)
            sf_sample = flatten(n_sample)
            samples.append(sf_sample)
        u = torch.stack(tuple(samples), dim=dim)
        return u
    else:
        return flatten(sample)

def augment(sample,dim=1,augment=True):
    samples = list()
    nvars = sample.shape[1]
    if augment:
        replicator = list()
        for i in range(nvars-1):
            replicator.append(flatten(sample))
        rep = torch.cat(tuple(replicator), dim=0)
        samples.append(rep)
    else:
        samples.append(flatten(sample))

    sf_sample = sample
    sf_samples = list()

    if augment:
        for i in range(dim-1):
            for i in range(nvars-1):
                sf_sample = order_shuffle(sf_sample,dim=1)
                sf_samples.append(flatten(sf_sample))
            samples.append(torch.cat(tuple(sf_samples),dim=0))
        u = torch.stack(tuple(samples), dim=1)
    else:
        for i in range(dim-1):
            sf_sample = rand_shuffle(sample, dim=1)
            sf_sample = flatten(sf_sample)
            samples.append(sf_sample)
        u = torch.stack(tuple(samples), dim=1)
    return u


def reduce_dim(sample, type='pca'):
    pca = PCA(n_components=1)
    batch_size = sample.shape[0]
    u =[]
    for i in range(batch_size):
        data = torch.reshape(sample[i], (sample.shape[1], sample.shape[2]))
        data = data.permute(1,0)
        pca.fit(data)
        u.append(pca.transform(data))
    u = torch.tensor(u)
    u = u.permute(0,2,1)
    return u


if __name__ == '__main__':
    sample = torch.randn([1, 10, 3, 6])
    print(sample)
    u = enhance(sample, stack_size=3)
    print(u)
    # print(u+sample)
















