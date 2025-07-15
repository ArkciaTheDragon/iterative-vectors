from args import parse_args

args = parse_args()

import numpy as np
import torch
from baukit import TraceDict
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
fs_val_set = FewshotDataset(val_set, rare_threshold)
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


def extract_tv(tokens, io_mask):
    last_only_mask = last_only_io_mask(io_mask)
    with TraceDict(model, layers=args.tv_target_layers) as t:
        model(**tokens)
    tv = {}
    for k in args.tv_target_layers:
        out = t[k].output
        tv[k] = (out[0] if isinstance(out, tuple) else out)[last_only_mask]
    return tv


def apply_tv(tokens, io_mask, ans_mask, layer_name, layer_tv):
    last_only_mask = last_only_io_mask(io_mask)

    def editor(output):
        state = output[0] if isinstance(output, tuple) else output
        state[last_only_mask] = layer_tv
        return output

    with TraceDict(model, layers=[layer_name], edit_output=editor, retain_output=False):
        logits = model(**tokens)["logits"][:, :-1][ans_mask[:, 1:]]
    return logits


# TV extraction
tvs = []
for _ in tqdm(range(args.episodes), "TV extraction"):
    support, query = fs_train_set.random_episode(rng, args.ext_shot, True)
    sq_samples = convert_for_verbalizer(i2label, support, query)
    sq_token, _, sq_io_mask = verbalize(vb, sq_samples, device)
    tv = extract_tv(sq_token, sq_io_mask)
    tvs.append(tv)
tvs = {l: torch.stack([tv[l] for tv in tvs]).mean(0) for l in args.tv_target_layers}

# best layer search
layer_result = {}
with tqdm(
    desc="TV best layer search", total=len(args.tv_target_layers * args.episodes)
) as pbar:
    for layer_name in args.tv_target_layers:
        rng = np.random.RandomState(args.seed)
        pred_labels, true_labels = [], []
        for _ in range(args.episodes):
            support, query = fs_val_set.random_episode(rng, args.shot, True)
            sq_samples = convert_for_verbalizer(i2label, support, query)
            sq_token, sq_ans_mask, sq_io_mask = verbalize(vb, sq_samples, device)

            logits = apply_tv(
                sq_token, sq_io_mask, sq_ans_mask, layer_name, tvs[layer_name]
            )
            pred = logits[0][limit].argmax().item()
            pred_labels.append(pred)
            true_labels.append(query["label"][0])

            pbar.update()
        f1 = calc_f1_score(pred_labels, true_labels, args.tv_search_metric)
        layer_result[layer_name] = f1
best_layer = max(layer_result, key=layer_result.get)
print(f"Best layer: {best_layer} ({layer_result[best_layer] * 100:.2f})")

# evaluation
pred_labels, true_labels = [], []
for _ in tqdm(range(args.eval_episodes), "TV eval"):
    support, query = fs_test_set.random_episode(rng, args.shot, True)
    sq_samples = convert_for_verbalizer(i2label, support, query)
    sq_token, sq_ans_mask, sq_io_mask = verbalize(vb, sq_samples, device)

    logits = apply_tv(sq_token, sq_io_mask, sq_ans_mask, best_layer, tvs[best_layer])
    edit_ans = logits[0][limit].argmax().item()
    pred_labels.append(edit_ans)
    true_labels.append(query["label"][0])

# print and record results
f1 = calc_f1_score(pred_labels, true_labels)
print("TV edit result:", "\t".join(f"{k}={v * 100:.2f}" for k, v in f1.items()))
with open(args.tv_result_fn, "a") as f:
    f.write(f"task={args.vtask} ")
    f.write(f"ext_shot={args.ext_shot} ")
    f.write(f"shot={args.shot} ")
    f.write(f"episodes={args.episodes} ")
    f.write(f"eval_episodes={args.eval_episodes} ")
    f.write(f"model={args.model_name} ")
    f.write(f"seed={args.seed} ")
    f.write(f"best_layer={best_layer} ")
    f.write(f"best_layer_f1={layer_result[best_layer] * 100:.2f} ")
    f.write(f"{' '.join(f'{k}={v * 100:.2f}' for k, v in f1.items())}")
    f.write("\n")
