import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    # model_params = 0
    # for parameter in model.parameters():
    #     model_params += parameter.numel()
    #     print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
    #     # print('Flops:' + flops)
    #     # print('Params:' + params)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class EarlyStopping_nri:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def masked_softmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    对输入张量执行掩码 Softmax，忽略零元素的计算并保留梯度流。

    Args:
        tensor (torch.Tensor): 输入张量，形状为 [..., seq_len]

    Returns:
        torch.Tensor: 掩码 Softmax 结果，零元素位置输出为 0
    """
    # 1. 创建掩码标识零元素位置（无需修改原始张量）
    zero_mask = (tensor == 0)

    # 2. 生成掩码后的张量（非原地操作）
    masked_tensor = torch.where(
        zero_mask,
        torch.tensor(float('-inf'), device=tensor.device),  # 零元素设为 -inf
        tensor  # 非零元素保留原值
    )

    # 3. 计算标准 Softmax
    softmax_out = torch.nn.functional.softmax(masked_tensor, dim=-1)

    # 4. 恢复零元素位置的值（非原地操作）
    result = torch.where(
        zero_mask,
        torch.tensor(0.0, device=tensor.device),  # 强制零元素输出为 0
        softmax_out
    )

    return result


# 从Gumbel(0, 1)分布中采样
def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device='cuda:0')
    return - torch.log(eps - torch.log(U + eps))

# 从Gumbel-Softmax分布中采样
def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / tau, dim=-1)

# Gumbel-Softmax函数
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.max(dim=-1)
        y_hard = torch.zeros_like(logits).scatter_(-1, k.unsqueeze(-1), 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

# 计算与均匀分布的分类KL散度
def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds+eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def calculate_avg_corr(corr_matrix, related_vars):
    """
    计算每个批次中的每个变量与相关变量的平均相关度。

    :param corr_matrix: [B, num_vars, num_vars] 相关系数矩阵 (batch维度)
    :param related_vars: 每个变量的相关变量索引 (list of lists, 每个批次和每个变量对应的相关变量序号)
    :return: [B, num_vars] 每个变量在每个批次中与其相关变量的平均相关度
    """
    B, num_vars, _ = corr_matrix.shape
    avg_corr = torch.zeros(B, num_vars)

    # 遍历每个批次
    for b in range(B):
        # 遍历每个变量
        related_list = related_vars[b]
        for i in range(num_vars):
             # 获取与变量i相关的变量序号列表
            related_indices = related_list[i]
            related_corrs = [corr_matrix[b, i, j].item() for j in related_indices]  # 获取该批次中相关变量的相关度值
            avg_corr[b, i] = torch.mean(torch.tensor(related_corrs))  # 计算该批次中该变量的平均相关度

    return avg_corr

def print_lag_variete(corr):
    # 初始化每个批次的结果列表
    batch_size, num_rows, num_cols = corr.size()
    all_indices_list = []
    global_max_len = 0

    # 首先计算所有批次中的全局最大长度
    for b in range(batch_size):
        nonzero_indices = corr[b].nonzero(as_tuple=False)
        indices_list = [[] for _ in range(num_rows)]

        for idx in nonzero_indices:
            row, col = idx
            indices_list[row.item()].append(col.item())

        max_len = max(len(indics) for indics in indices_list)
        global_max_len = max(global_max_len, max_len)

    # 生成填充后的结果列表
    for b in range(batch_size):
        nonzero_indices = corr[b].nonzero(as_tuple=False)
        indices_list = [[] for _ in range(num_rows)]

        for idx in nonzero_indices:
            row, col = idx
            indices_list[row.item()].append(col.item())

        # 填充每个列表到全局最大长度
        for i, indices in enumerate(indices_list):
            indices_list[i] = indices + [i] * (global_max_len - len(indices))

        all_indices_list.append(indices_list)

    return all_indices_list