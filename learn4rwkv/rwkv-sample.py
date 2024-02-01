"""
origin code from https://johanwind.github.io/2023/03/23/rwkv_details.html
LBJ do some trying in it
"""
import numpy as np
from torch import load as torch_load  # Only for loading the model weights
from tokenizers import Tokenizer

def simple_layer_norm(x, w, b):
    """
    对输入数据 x 进行 Z-Score 标准化(标准正态分布)，并乘以权重 w 和加上偏置 b。

    参数:
    x: 输入数据，可以是 NumPy 数组或列表。
    w: 权重，用于缩放标准化后的数据。
    b: 偏置，用于调整标准化数据的中心位置。

    返回:
    标准化后的数组。
    """
    # 计算均值和标准差
    mean_x = np.mean(x)
    std_x = np.std(x)
    
    # 应用 Z-Score 标准化公式
    normalized_x = (x - mean_x) / std_x * w + b
    
    return normalized_x

def simple_params(prefix, model):
    """
    从模型中提取所有以特定前缀开头的参数。

    参数:
    prefix: 要匹配的键的前缀。
    model: 包含模型参数的字典。

    返回:
    一个列表，包含所有以给定前缀开头的键对应的值。
    """
    return [model[key] for key in model.keys() if key.startswith(prefix)]

layer_norm = lambda x, w, b : (x - np.mean(x)) / np.std(x) * w + b
exp = np.exp
sigmoid = lambda x : 1/(1 + exp(-x))

def time_mixing(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout):
    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    v = Wv @ ( x * mix_v + last_x * (1 - mix_v) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )

    wkv = (last_num + exp(bonus + k) * v) /      \
          (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x,num,den)


def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )
    vk = Wv @ np.maximum(k, 0)**2
    return sigmoid(r) * vk, x


def RWKV(model, token, state):
    # 读取模型参数
    params = lambda prefix : [model[key] for key in model.keys() if key.startswith(prefix)]

    # 读取词嵌入向量
    x = params('emb')[0][token]
    # x = layer_norm(x, *params('blocks.0.ln0'))
    x = simple_layer_norm(x, *params('blocks.0.ln0'))
    print(x.shape)

    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f'blocks.{i}.ln1'))
        dx, state[i][:3] = time_mixing(x_, *state[i][:3], *params(f'blocks.{i}.att'))
        x = x + dx

        x_ = layer_norm(x, *params(f'blocks.{i}.ln2'))
        dx, state[i][3] = channel_mixing(x_, state[i][3], *params(f'blocks.{i}.ffn'))
        x = x + dx

    x = layer_norm(x, *params('ln_out'))
    x = params('head')[0] @ x

    e_x = exp(x-np.max(x))
    probs = e_x / e_x.sum() # Softmax of x

    return probs, state

##########################################################################################################
# 简单地对概率分布进行采样（在实践中，我们避免了低概率标记），并在文本中添加另一个标记。然后，我们将新令牌输入RWKV并重复。
def sample_probs(probs, temperature=1.0, top_p=0.85):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs**(1/temperature)
    return np.random.choice(a=len(probs), p=probs/np.sum(probs))


# Available at https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth
MODEL_FILE = '/home/lbj/桌面/models/learn/RWKV-4-Pile-430M-20220808-8066.pth'
N_LAYER = 24
N_EMBD = 1024

print(f'\nLoading {MODEL_FILE}')
weights = torch_load(MODEL_FILE, map_location='cpu')
for k in weights.keys():
    if '.time_' in k: weights[k] = weights[k].squeeze()
    weights[k] = weights[k].float().numpy() # convert to f32 type


# Available at https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json
tokenizer = Tokenizer.from_file("math4science/learn4rwkv/tokenizer/20B_tokenizer.json")

print(f'\nPreprocessing context')

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# context = "道可道，非常道"

# token为输入文本的状态表示，变量“probs”包含模型预测下一个标记的概率分布。
state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)

print(context, end="")
for i in range(100):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end="", flush=True)
    probs, state = RWKV(weights, token, state)