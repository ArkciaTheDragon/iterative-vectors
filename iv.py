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


def extract_iv(full_token, ans_mask, shift=True, mask_interm_ans=False):
    shift_ans_mask = ans_mask[:, 1:] if shift else ans_mask
    attn_mask = full_token["attention_mask"]
    if mask_interm_ans:
        attn_mask = full_token["attention_mask"].clone()
        attn_mask[:, 1:] &= ~ans_mask[:, :-1]
    with TraceDict(model, layers=args.target_layers) as t:
        logits = model(input_ids=full_token["input_ids"], attention_mask=attn_mask)[
            "logits"
        ]
        shift_logits = logits[:, :-1] if shift else logits
        ans_logits = shift_logits[shift_ans_mask]
        iv = {}
        for k in args.target_layers:
            shift_out = t[k].output[:, :-1] if shift else t[k].output
            iv[k] = shift_out[shift_ans_mask]
    return ans_logits, iv


def apply_iv(full_token, io_mask, ans_mask, iv, strength=1):
    shift_ans_mask = ans_mask[:, 1:]

    def editor(output, layer, input):
        output[io_mask] += strength * iv[layer]
        return output

    with TraceDict(
        model, layers=args.target_layers, edit_output=editor, retain_output=False
    ):
        logits = model(**full_token)["logits"][:, :-1][shift_ans_mask]
    return logits


def extract_and_apply_iv(
    full_token,
    io_mask,
    ans_mask,
    iv,
    support_label,
    strength,
    shift=False,
    mask_interm_ans=False,
):
    shift_ans_mask = ans_mask[:, 1:] if shift else ans_mask
    attn_mask = full_token["attention_mask"]
    if iv is not None:
        iv_to_add = {
            l: torch.cat(
                (
                    v.gather(0, support_label.unsqueeze(1).expand(-1, v.size(1))),
                    v[-1:],
                ),
            )
            for l, v in iv.items()
        }

    if mask_interm_ans:
        attn_mask = full_token["attention_mask"].clone()
        attn_mask[:, 1:] &= ~ans_mask[:, :-1]

    def editor(output, layer, input):
        if iv is not None:
            output[io_mask] += strength * iv_to_add[layer]
        return output

    with TraceDict(model, layers=args.target_layers, edit_output=editor) as t:
        logits = model(input_ids=full_token["input_ids"], attention_mask=attn_mask)[
            "logits"
        ]
        shift_logits = logits[:, :-1] if shift else logits
        ans_logits = shift_logits[shift_ans_mask]

        iv_out = {}
        for k in args.target_layers:
            shift_out = t[k].output[:, :-1] if shift else t[k].output
            iv_out[k] = shift_out[io_mask]
    return ans_logits, iv_out


# Extraction
clean_result, edit_result = [], []
extracted = {l: [] for l in args.target_layers}
ivs = None
if not args.run_clean:
    for ext_ep in tqdm(range(args.episodes), "Extraction"):
        support, query = fs_train_set.random_episode(rng, args.ext_shot, False)

        s_samples = convert_for_verbalizer(i2label, support)
        s_label = torch.tensor(support["label"], device=device)
        q_sample = convert_for_verbalizer(i2label, query)[0]

        reord = torch.randperm(len(s_samples))
        _, ord_back = reord.sort()
        re_label = s_label[reord]
        pos_sample = [s_samples[i] for i in reord]
        pos_sample.append(q_sample)
        neg_sample = [q_sample]

        pos_token, pos_ans_mask, pos_io_mask = verbalize(vb, pos_sample, device)
        neg_token, neg_ans_mask, neg_io_mask = verbalize(vb, neg_sample, device)
        _, pos_ivs = extract_and_apply_iv(
            pos_token, pos_io_mask, pos_ans_mask, ivs, re_label, args.ext_strength
        )
        _, neg_ivs = extract_iv(neg_token, neg_io_mask)
        for l in args.target_layers:
            diff = pos_ivs[l] - neg_ivs[l]
            cls_ivs = []
            for cls in range(len(first_tokens)):
                cls_ivs.append(diff[:-1][re_label == cls].mean(0))
            cls_ivs.append(diff[-1])
            extracted[l].append(torch.stack(cls_ivs))

        if (
            args.ext_batch
            and ext_ep % args.ext_batch == 0
            or ext_ep == args.episodes - 1
        ):
            ivs = {l: torch.stack(x).mean(0) for l, x in extracted.items()}

# Evaluation
true_labels = []
target_set = fs_test_set if args.run_clean or args.run_test else fs_val_set
for _ in tqdm(range(args.eval_episodes), "Eval"):
    support, query = target_set.random_episode(
        rng, args.shot, args.run_clean or args.shot > 1
    )
    sq_samples = convert_for_verbalizer(i2label, support, query)
    sq_token, sq_ans_mask, sq_io_mask = verbalize(vb, sq_samples, device)

    true_labels.append(query["label"][0])
    if args.run_clean:
        clean_logits = model(**sq_token)["logits"][:, :-1][sq_ans_mask[:, 1:]]
        clean_ans = clean_logits[0][limit].argmax().item()
        clean_result.append(clean_ans)
    else:
        s_labels = torch.tensor(support["label"], device=device)
        ep_iv = {l: torch.cat((v[s_labels], v[-1:]), dim=0) for l, v in ivs.items()}

        edit_logits = apply_iv(sq_token, sq_io_mask, sq_ans_mask, ep_iv, args.strength)
        edit_ans = edit_logits[0][limit].argmax().item()
        edit_result.append(edit_ans)

# Print and record results
if args.run_clean:
    f1 = calc_f1_score(clean_result, true_labels)
    print("Clean result:", "\t".join(f"{k}={v * 100:.2f}" for k, v in f1.items()))
    with open(args.clean_result_fn, "a") as f:
        f.write(
            f"task={args.vtask} shot={args.shot} "
            f"eval_episodes={args.eval_episodes} model={args.model_name} "
        )
        f.write(f"{' '.join(f'{k}={v * 100:.2f}' for k, v in f1.items())}")
        f.write("\n")
else:
    save_fn = args.test_result_fn if args.run_test else args.edit_result_fn
    f1 = calc_f1_score(edit_result, true_labels)
    print("Edit result:", "\t".join(f"{k}={v * 100:.2f}" for k, v in f1.items()))
    with open(save_fn, "a") as f:
        f.write(f"task={args.vtask} ")
        f.write(f"model={args.model_name} ")
        f.write(f"shot={args.shot} ")
        f.write(f"ext_shot={args.ext_shot} ")
        f.write(f"strength={args.strength} ")
        f.write(f"ext_strength={args.ext_strength} ")
        f.write(f"ext_batch={args.ext_batch} ")
        f.write(f"episodes={args.episodes} ")
        f.write(f"eval_episodes={args.eval_episodes} ")
        f.write(f"seed={args.seed} ")
        f.write(f"{' '.join(f'{k}={v * 100:.2f}' for k, v in f1.items())}")
        f.write("\n")
