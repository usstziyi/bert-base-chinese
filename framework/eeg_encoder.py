import torch
from braindecode.models import EEGNet


# ============================================================
# 1. EEG 参数
# ============================================================

batch_size = 16

n_chans = 64        # EEG 通道数
sfreq = 250         # 采样率 Hz
duration = 4        # 每个样本 4 秒

n_times = sfreq * duration

n_classes = 4       # 分类类别数


# ============================================================
# 2. 创建 EEGNet
# ============================================================

model = EEGNet(
    n_chans=n_chans,
    n_outputs=n_classes,
    n_times=n_times,
    sfreq=sfreq,
)

print(model)


# ============================================================
# 3. 构造一批模拟 EEG
#
# shape:
# (batch_size, n_chans, n_times)
# ============================================================

x = torch.randn(
    batch_size,
    n_chans,
    n_times,
)

y = torch.randint(
    low=0,
    high=n_classes,
    size=(batch_size,),
)


print("\nEEG input shape:")
print(x.shape)

print("\nlabel shape:")
print(y.shape)


# ============================================================
# 4. 前向传播
# ============================================================

logits = model(x)

print("\nmodel output shape:")
print(logits.shape)



# ============================================================
# 5. 计算分类损失
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

loss = criterion(
    logits,
    y,
)

print("\nloss:")
print(loss.item())




# ============================================================
# 6. 反向传播
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nTraining step finished.")