"""
常量定义文件，包含序列处理和模型用到的特殊标记
"""

# 特殊标记的索引
PAD = 0  # 填充标记
EOS = 1  # 序列结束标记
BOS = 2  # 序列开始标记
UNK = 3  # 未知节点标记

# 特殊标记的字符串表示
PAD_WORD = '<blank>'
EOS_WORD = '</s>'
BOS_WORD = '<s>'
UNK_WORD = '<unk>'

# 模型参数默认值
DEFAULT_EMBED_DIM = 128  # 嵌入维度默认值
DEFAULT_HIDDEN_SIZE = 256  # 隐藏层大小默认值
