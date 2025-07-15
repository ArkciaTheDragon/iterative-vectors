from args import parse_args

args = parse_args()

import numpy as np
import torch
from baukit import TraceDict, get_module
from einops import rearrange
from tqdm import tqdm

from util import (
    FewshotDataset,
    Verbalizer,
    calc_f1_score,
    convert_for_verbalizer,
    get_i2label,
    last_only_io_mask,
    load_dataset_splits,
    load_tokenizer_and_model,
    verbalize,
)

torch.random.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)

# load dataset
train_set, val_set, test_set = load_dataset_splits(args.dataset, args.task)
rare_threshold = 0
fs_train_set = FewshotDataset(train_set, rare_threshold)
fs_test_set = FewshotDataset(test_set, rare_threshold)

# load model
print(f"Loading model {args.model}...")
tokenizer, model = load_tokenizer_and_model(args)
model = model.eval()
model.requires_grad_(False)
device = model.device

# verbalizer
vb = Verbalizer(tokenizer)
i2label = get_i2label(args.vtask, train_set)
first_tokens = {
    i: tokenizer.encode(l, add_special_tokens=False)[0] for i, l in i2label.items()
}
limit = torch.tensor(list(first_tokens.values()), dtype=torch.long, device=device)


def split_act_by_heads(activation):
    n_heads = model.config.num_attention_heads
    return rearrange(activation, "... (n h) -> ... n h", n=n_heads)


def extract_attn_act(token, ans_mask):
    shift_ans_mask = ans_mask[:, 1:]
    attn_layers = []
    with TraceDict(
        model, layers=args.target_layers, retain_input=True, retain_output=False
    ) as t:
        model(**token)
    for k in args.target_layers:
        shift_input = t[k].input[:, :-1]
        attn = shift_input[shift_ans_mask]  # [n_ans_token, hid_dim]
        split_attn = split_act_by_heads(attn[0])  # [heads, head_dim]
        attn_layers.append(split_attn)
    return torch.stack(attn_layers)  # [layer, heads, head_dim]


def o_project(layer_name: str, inp: torch.Tensor):
    proj_module = get_module(model, layer_name)
    o_proj_w = proj_module.weight
    if "llama" in args.model_name or "gpt-j" in args.model_name:
        out = torch.matmul(inp, o_proj_w.T)
    elif "gpt-neox" in args.model_name:
        out_proj_bias = proj_module.bias
        out = torch.addmm(out_proj_bias, inp.squeeze(), o_proj_w.T)
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")
    return out


def get_mean_head_act_adder(mean_head_acts, layer_name: str, layer: int, head: int):

    def editor(output, current_layer, inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = split_act_by_heads(inp)  # [batch, seq_len, heads, head_dim]
        inp[:, :, head] = mean_head_acts[layer, head]
        inp = rearrange(inp, "... n h -> ... (n h)")
        output = o_project(layer_name, inp)
        return output

    return editor


def get_function_vector_editor(function_vector, io_mask):
    last_only_mask = last_only_io_mask(io_mask)

    def editor(output):
        output[last_only_mask] += function_vector
        return output

    return editor


# Step 1: get mean head activations
activations = []
extracted = {l: [] for l in args.target_layers}
for _ in tqdm(range(args.episodes), "FV attn activations"):
    support, query = fs_train_set.random_episode(rng, args.ext_shot, False)
    sq_samples = convert_for_verbalizer(i2label, support, query)
    sq_token, sq_ans_mask, _ = verbalize(vb, sq_samples, device)

    attn_acts = extract_attn_act(sq_token, sq_ans_mask)  # [layer, heads, head_dim]
    activations.append(attn_acts)
mean_head_act = torch.stack(activations).mean(0)  # [layer, heads, head_dim]
mean_head_act = mean_head_act.to(device, model.dtype)

# Step 2: calc indirect effect values
indirect_effects = torch.zeros(
    args.ie_episodes, len(args.target_layers), model.config.num_attention_heads
)
rng = np.random.RandomState(args.seed)  # reset seed to ensure same sample sequence
total_cnt = (
    args.ie_episodes * len(args.target_layers) * model.config.num_attention_heads
)
with tqdm(desc="FV indirect effect", total=total_cnt) as pbar:
    for ep in range(args.ie_episodes):
        support, query = fs_train_set.random_episode(rng, args.ext_shot, False)
        q_token_id = first_tokens[query["label"][0]]

        # clean run
        sq_samples = convert_for_verbalizer(i2label, support, query)
        sq_token, sq_ans_mask, _ = verbalize(vb, sq_samples, device)
        clean_logits = model(**sq_token).logits
        clean_ans = clean_logits[:, :-1][sq_ans_mask[:, 1:]][0]
        clean_probs = torch.softmax(clean_ans, dim=-1)

        # substitute run
        shuf_label = torch.tensor(support["label"], device=device)
        shuf_label = shuf_label[torch.randperm(len(shuf_label))].tolist()
        shuf_support = {
            k: v if k != "label" else shuf_label for k, v in support.items()
        }
        sq_samples = convert_for_verbalizer(i2label, shuf_support, query)
        sq_token, sq_ans_mask, _ = verbalize(vb, sq_samples, device)
        for layer, layer_name in enumerate(args.target_layers):
            for head in range(model.config.num_attention_heads):
                editor = get_mean_head_act_adder(mean_head_act, layer_name, layer, head)
                with TraceDict(model, layers=[layer_name], edit_output=editor):
                    logits = model(**sq_token).logits
                subs_logits = logits[:, :-1][sq_ans_mask[:, 1:]][0]
                subs_probs = torch.softmax(subs_logits, dim=-1)

                # indirect effect on the true label token
                e = (subs_probs - clean_probs).squeeze()
                indirect_effects[ep, layer, head] = e[q_token_id]

                pbar.update()

# Step 3: select top-n heads
mean_effect = indirect_effects.mean(0)  # [layer, head]
topk_vals, topk_inds = torch.topk(mean_effect.flatten(), k=args.n_top_heads)
top_heads = list(
    zip(*np.unravel_index(topk_inds, mean_effect.shape), topk_vals.tolist())
)  # [(layer, head, effect)]

# Step 4: compute function vectors
function_vector = torch.zeros(model.config.hidden_size, device=device)
for layer, head, _ in top_heads:
    head_size = model.config.hidden_size // model.config.num_attention_heads
    x = torch.zeros(model.config.hidden_size, device=device, dtype=model.dtype)
    x[head * head_size : (head + 1) * head_size] = mean_head_act[layer, head]
    function_vector += o_project(args.target_layers[layer], x)
function_vector = function_vector.to(device, model.dtype).unsqueeze(0)

# Finally, evaluation
true_labels, edit_result = [], []
target_set = fs_test_set
layer = model.config.num_hidden_layers // 3  # add at |L|/3
target_layer = args.target_layers[layer - 1 : layer]
for _ in tqdm(range(args.eval_episodes), "FV eval"):
    support, query = target_set.random_episode(rng, args.shot, args.run_clean)
    sq_samples = convert_for_verbalizer(i2label, support, query)
    sq_token, sq_ans_mask, sq_io_mask = verbalize(vb, sq_samples, device)

    editor = get_function_vector_editor(function_vector, sq_io_mask)
    with TraceDict(model, layers=target_layer, edit_output=editor):
        logits = model(**sq_token).logits
    edit_logits = logits[:, :-1][sq_ans_mask[:, 1:]]

    true_labels.append(query["label"][0])
    edit_ans = edit_logits[0][limit].argmax().item()
    edit_result.append(edit_ans)

save_fn = args.fv_result_fn
f1 = calc_f1_score(edit_result, true_labels)
print("FV edit result:", "\t".join(f"{k}={v * 100:.2f}" for k, v in f1.items()))
with open(save_fn, "a") as f:
    f.write(f"task={args.vtask} ")
    f.write(f"ext_shot={args.ext_shot} ")
    f.write(f"shot={args.shot} ")
    f.write(f"episodes={args.episodes} ")
    f.write(f"ie_episodes={args.ie_episodes} ")
    f.write(f"eval_episodes={args.eval_episodes} ")
    f.write(f"model={args.model_name} ")
    f.write(f"seed={args.seed} ")
    f.write(f"n_top_heads={args.n_top_heads} ")
    f.write(f"{' '.join(f'{k}={v * 100:.2f}' for k, v in f1.items())}")
    f.write("\n")
