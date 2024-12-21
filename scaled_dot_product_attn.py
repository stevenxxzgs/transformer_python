# 1. 防止梯度消失/爆炸：

# 点积的方差： 假设查询 q 和键 k 的每个分量都是独立同分布的，均值为 0，方差为 1。那么，点积 q·k 的方差将是 d_k（键的维度）。这是因为点积是 d_k 个独立随机变量的和，而独立随机变量的和的方差等于它们方差的和。
# 方差过大的问题： 当 d_k 很大时，点积的方差也会很大。这会导致点积的数值分布变得很宽，一些值会很大，一些值会很小。
# Softmax 的敏感区间： Softmax 函数在输入值差异很大时，输出会趋向于 one-hot 向量（只有一个值为 1，其余为 0）。这意味着梯度会变得非常小，接近于 0，从而导致梯度消失，使得模型难以训练。相反，如果输入值过小，softmax的梯度又会变得很大，导致梯度爆炸。
# 缩放的作用： 通过除以 sqrt(d_k)，可以将点积的方差缩放到 1，从而使 softmax 的输入值分布更加合理，落在 softmax 的敏感区域内，避免梯度消失或爆炸的问题。
# 2. 提高训练的稳定性：

# 缩放点积注意力有助于提高训练的稳定性。如果没有缩放，模型在训练初期可能会因为梯度过大或过小而难以收敛。缩放可以使梯度更加稳定，从而加快收敛速度并提高最终模型的性能。

# Q：为什么softmax有这样的特性？
# A：softmax函数在输入值差异很大时，输出会趋向于 one-hot 向量（只有一个值为 1，其余为 0）。这意味着梯度会变得非常小，接近于 0，从而导致梯度消失，使得模型难以训练。相反，如果输入值过小，softmax的梯度又会变得很大，导致梯度爆炸。

import torch
import math
import numpy as np
import matplotlib.pyplot as plt

d_k = 64  # 键的维度
batch_size = 3  # 批次大小
num_queries = 30  # query 的数量
head = 8

q = torch.randn(batch_size, head, num_queries, d_k)  # 修改为多个query
k = torch.randn(batch_size, head, num_queries, d_k)  # 修改为多个key

# 不缩放
attn_unscaled = torch.matmul(q, k.transpose(2, 3))
probs_unscaled = torch.softmax(attn_unscaled, dim=-1)

# 缩放
attn_scaled = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)
probs_scaled = torch.softmax(attn_scaled, dim=-1)

print("Unscaled Attention Probabilities (Example - First batch, first head):\n", probs_unscaled[0, 0, :5, :5])  # 只打印部分，避免输出过多
print("\nScaled Attention Probabilities (Example - First batch, first head):\n", probs_scaled[0, 0, :5, :5])  # 只打印部分

# 计算方差进行验证 (针对每个批次的每个head的注意力得分矩阵)
attn_unscaled_np = attn_unscaled.detach().numpy()  # 需要detach，否则会报错
attn_scaled_np = attn_scaled.detach().numpy()

unscaled_variances = []
scaled_variances = []

for i in range(batch_size):
    for h in range(head):
        unscaled_variance = np.var(attn_unscaled_np[i, h])
        scaled_variance = np.var(attn_scaled_np[i, h])
        unscaled_variances.append(unscaled_variance)
        scaled_variances.append(scaled_variance)
        print(f"\nBatch {i + 1}, Head {h+1}:")
        print("Unscaled Attention Variance:", unscaled_variance)
        print("Scaled Attention Variance:", scaled_variance)

# 计算所有批次和head的平均方差
avg_unscaled_variance = np.mean(unscaled_variances)
avg_scaled_variance = np.mean(scaled_variances)
print("\nAverage Unscaled Attention Variance:", avg_unscaled_variance)
print("Average Scaled Attention Variance:", avg_scaled_variance)

# 绘制直方图以可视化分布
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(attn_unscaled_np.flatten(), bins=50, alpha=0.7, label='Unscaled')
plt.title('Distribution of Unscaled Attention Scores')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(attn_scaled_np.flatten(), bins=50, alpha=0.7, label='Scaled')
plt.title('Distribution of Scaled Attention Scores')
plt.legend()

plt.show()

#模拟softmax输入差异导致梯度不同的情况
def simulate_softmax_gradient(x):
    y = torch.softmax(x, dim=-1)
    grad = torch.autograd.grad(y.sum(), x)[0]
    return grad

x_small_diff = torch.tensor([-1.0, -0.9, -0.8], requires_grad=True)
x_large_diff = torch.tensor([-10.0, -5.0, 0.0], requires_grad=True)

grad_small = simulate_softmax_gradient(x_small_diff)
grad_large = simulate_softmax_gradient(x_large_diff)

print("\nSimulating Softmax Gradient:")
print("Gradient with small input differences:", grad_small)
print("Gradient with large input differences:", grad_large)