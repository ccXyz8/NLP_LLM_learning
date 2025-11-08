import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
D_MODEL = 512  # 模型维度
H = 8  # 多头注意力头数
D_FF = 2048  # 前馈网络维度
DROPOUT = 0.1  # dropout率
N_LAYERS = 6  # 编码器和解码器层数

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 100

# 数据参数
MAX_LENGTH = 100  # 最大序列长度
PAD_TOKEN_ID = 0  # 填充token的ID

# 词汇表大小（根据实际数据调整）
SRC_VOCAB_SIZE = 10000
TGT_VOCAB_SIZE = 10000