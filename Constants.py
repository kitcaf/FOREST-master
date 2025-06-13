"""
常量定义文件，包含序列处理和模型用到的特殊标记和参数
"""

# 特殊标记的索引
PAD = 0  # 填充标记
EOS = 1  # 序列结束标记
BOS = 2  # 序列开始标记
UNK = 3  # 未知节点标记

# 用户ID起始索引
USER_START_IDX = 4  # 用户ID从4开始，前面是特殊标记

# 特殊标记的字符串表示
PAD_WORD = '<blank>'
EOS_WORD = '</blank>'
BOS_WORD = ' '
UNK_WORD = ' '

# 模型参数默认值
DEFAULT_EMBED_DIM = 128  # 嵌入维度默认值
DEFAULT_HIDDEN_SIZE = 256  # 隐藏层大小默认值

# 束搜索参数
DEFAULT_BEAM_SIZE = 5  # 束宽默认值
DEFAULT_TEMPERATURE = 1.0  # 温度采样参数默认值

# 社交网络修正参数
SOCIAL_ALPHA = 0.3  # 社交修正权重
SOCIAL_ALPHA_POS1 = 0.3  # 第一个位置的社交修正权重
SOCIAL_ALPHA_POS2 = 0.2  # 第二个位置的社交修正权重
SOCIAL_ALPHA_POS3 = 0.1  # 第三个位置的社交修正权重

# 噪声注入率
MIN_NOISE_RATIO = 0.0  # 最小噪声注入率
MAX_NOISE_RATIO = 0.15  # 最大噪声注入率

# 评估指标参数
DEFAULT_K_VALUES = [10, 50, 100]  # 默认的K值列表，用于Hits@K和MAP@K

# 输出目录
OUTPUT_DIR = "output"  # 输出文件保存目录
MODELS_DIR = "models"  # 模型保存目录

# 最大序列长度
MAX_SEQ_LEN = 20  # 最大序列长度

# 最大预测长度
MAX_PRED_LENGTH = 5  # 最大预测长度

# 数值稳定性
EPS = 1e-8  # 数值稳定性小常数
