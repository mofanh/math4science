########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method


class RWKV_TOKENIZER:
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]

    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []  # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(
            range(len(sorted))
        ):  # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode("utf-8")

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except:
                pass
            print(f"{repr(s)}{i}", end=" ")
            # print(repr(s), i)
        print()


########################################################################################################


def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


########################################################################################################

# tokenizer = RWKV_TOKENIZER("/home/lbj/桌面/math4science/learn4rwkv/tokenizer/rwkv_vocab_v20230424.txt")
tokenizer = RWKV_TOKENIZER(
    r"G:\math4science\learn4rwkv\tokenizer\rwkv_vocab_v20230424.txt"
)

# THIS IS NOW UPDATED TO SUPPORT LATEST RWKV-5 WORLD v2 MODELS

args = types.SimpleNamespace()
args.MODEL_NAME = r"G:\models\RWKV-5-World-0.4B-v2-20231113-ctx4096"
# args.MODEL_NAME = '/home/lbj/桌面/models/learn/RWKV-5-World-0B4-v2-OnlyForTest_71%_trained-20231104-ctx4096'
# args.MODEL_NAME = 'models/RWKV-5-World-0B4-v2-OnlyForTest_71%_trained-20231104-ctx4096'
args.n_layer = 24
args.n_embd = 1024
args.vocab_size = 65536

# context = "\nElon Musk has"
context = "\nEnglish and chinese and Janpanese are language"
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.7


class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval()  # set torch to inference mode

        w = torch.load(args.MODEL_NAME + ".pth", map_location="cpu")
        for k in w.keys():
            w[k] = w[k].float()  # convert to f32 type
            if ".time_" in k:
                w[k] = w[k].squeeze()
            if ".time_decay" in k:
                w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
            if ".time_faaaa" in k:
                w[k] = w[k].unsqueeze(-1)

        self.n_head = w["blocks.0.att.time_decay"].shape[0]
        self.head_size = w["blocks.0.ln1.weight"].shape[0] // self.n_head

        self.w = types.SimpleNamespace()  # set self.w from w
        self.w.blocks = {}
        for (
            k
        ) in (
            w.keys()
        ):  # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split(".")
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here:
                        here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p):
                        setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        i0 = (2 + self.head_size) * i + 0
        xk = x * time_mix_k + state[i0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[i0] * (1 - time_mix_r)
        state[i0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    @MyFunction
    def time_mixing(
        self,
        x,
        state,
        i: int,
        time_mix_k,
        time_mix_v,
        time_mix_r,
        time_mix_g,
        time_first,
        time_decay,
        kw,
        vw,
        rw,
        gw,
        ow,
        ln_w,
        ln_b,
    ):
        H = self.n_head
        S = self.head_size

        i1 = (2 + S) * i + 1
        xk = x * time_mix_k + state[i1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[i1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[i1] * (1 - time_mix_r)
        xg = x * time_mix_g + state[i1] * (1 - time_mix_g)
        state[i1] = x

        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg)

        s = state[(2 + S) * i + 2 : (2 + S) * (i + 1), :].reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + time_decay * s

        state[(2 + S) * i + 2 : (2 + S) * (i + 1), :] = s.reshape(S, -1)
        x = x.flatten()

        x = (
            F.group_norm(
                x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5
            ).squeeze(0)
            * g
        )  # same as gn(x/8, eps=1e-5)
        return ow @ x

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(
                    self.args.n_layer * (2 + self.head_size), self.args.n_embd
                )

            x = self.w.emb.weight[token]
            # plt.subplot(2, 3, 1)
            # plt.title("emb.weight")
            # plt.plot(x)

            x = self.layer_norm(x, self.w.blocks[0].ln0)
            # plt.subplot(2, 3, 2)
            # plt.title("layer_norm")
            # plt.plot(x)

            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln1),
                    state,
                    i,
                    att.time_mix_k,
                    att.time_mix_v,
                    att.time_mix_r,
                    att.time_mix_g,
                    att.time_faaaa,
                    att.time_decay,
                    att.key.weight,
                    att.value.weight,
                    att.receptance.weight,
                    att.gate.weight,
                    att.output.weight,
                    att.ln_x.weight,
                    att.ln_x.bias,
                )
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln2),
                    state,
                    i,
                    ffn.time_mix_k,
                    ffn.time_mix_r,
                    ffn.key.weight,
                    ffn.value.weight,
                    ffn.receptance.weight,
                )

            # plt.subplot(2, 3, 3)
            # plt.title(f"self.args.n_layer")
            # plt.plot(x)

            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)

            # plt.subplot(2, 3, 4)
            # plt.title("head.weight")
            # plt.imshow(self.w.head.weight)

            # plt.subplot(2, 3, 5)
            # plt.title("layer_norm(x, self.w.ln_out)")
            # # plt.imshow(self.layer_norm(x, self.w.ln_out))

            # plt.subplot(2, 3, 6)
            # plt.title("head.weight @ layer_norm(x, self.w.ln_out)")
            # plt.plot(x)
            return x.float(), state


print(f"\nUsing CPU. Loading {args.MODEL_NAME} ...")
model = RWKV_RNN(args)

print(f"\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)")
init_state = None

length = len(tokenizer.encode(context))
i = 1
old_state = torch.zeros(args.n_layer * (2 + 64), args.n_embd)
for token in tokenizer.encode(context):
    if init_state != None:
        old_state = init_state.clone()
    init_out, init_state = model.forward(token, init_state)

    # 计算均值和标准差
    mean = torch.mean(init_state)
    std = torch.std(init_state, unbiased=True)

    # 设置颜色映射
    cmap = plt.get_cmap("jet")

    plt.subplot(1, 2, 1)

    # 使用imshow函数展示矩阵
    plt.imshow(
        init_state,
        cmap=cmap,
        vmin=mean - 1 * std,
        vmax=mean + 1 * std,
    )
    # 设置子图标题
    plt.title(f"state{i}:{token}")

    plt.subplot(1, 2, 2)
    
    if old_state == None:
        plt.imshow(
            init_state,
            cmap=cmap,
            vmin=mean - 1 * std,
            vmax=mean + 1 * std,
        )
    else:
        # minus = init_state - old_state
        mean = torch.mean(init_state - old_state)
        std = torch.std(init_state - old_state, unbiased=True)
        plt.imshow(
            init_state - old_state,
            cmap=cmap,
            vmin=mean - 1 * std,
            vmax=mean + 1 * std,
        )

    # 添加颜色条
    # plt.colorbar()

    # 设置子图标题
    plt.title(f"state{i} - state{i-1}")

    # 保存图像
    # plt.savefig(f"/home/lbj/桌面/math4science/temp/state_{i}.png", dpi=3000, bbox_inches="tight")
    plt.savefig(f"temp\\state_{i}.png", dpi=3000, bbox_inches="tight")

    # 更新计数器
    i += 1

# 在循环结束后显示所有子图
# plt.show()

for TRIAL in range(NUM_TRIALS):
    print(f"\n\n--[ Trial {TRIAL} ]-----------------", context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if "\ufffd" not in tmp:  # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass
        out, state = model.forward(token, state)
        # plt.subplot(LENGTH_PER_TRIAL/10, LENGTH_PER_TRIAL/10, i+1)
        # plt.title(i+1)
        # # plt.imshow(out)
        # plt.plot(out)
    # plt.show()

print("\n")