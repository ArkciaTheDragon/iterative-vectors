import os
from pathlib import Path

import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from datasets import load_dataset, load_dataset_builder
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


def random_examples(rng, dataset, n):
    """Get n random examples from a dataset"""
    ids = rng.choice(len(dataset), n, replace=False)
    return dataset[ids]


def load_tokenizer_and_model(args) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model from args"""
    name_or_path = Path(args.model)
    if not name_or_path.exists():
        ans = input(f'Model "{name_or_path}" not found locally, continue? [y/N]: ')
        if ans.strip().lower() != "y":
            exit(0)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    dtype = torch.float32 if args.no_half else torch.bfloat16
    attn = "flash_attention_2" if args.flash_attention else None
    if args.device_map is None:  # load small model directly
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        if not args.no_half:
            model = model.half()
        model = model.to(args.gpu[0])
        return tokenizer, model
    elif isinstance(args.device_map, str):  # rely on predefined device map
        print(
            f"Using device map: {args.device_map} "
            f"with gpu {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            attn_implementation=attn,
            torch_dtype=dtype,
            device_map=args.device_map,
        )
    else:  # custom device map
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                attn_implementation=attn,
                torch_dtype=dtype,
            )
        if args.mem_limit:
            max_mem = {args.gpu[0]: f"{args.mem_limit}GiB"}
            for i in args.gpu[1:]:
                max_mem[i] = torch.cuda.get_device_properties(i).total_memory
        else:
            max_mem = {
                i: torch.cuda.get_device_properties(i).total_memory for i in args.gpu
            }
        dmap = infer_auto_device_map(
            model,
            max_memory=max_mem,
            no_split_module_classes=args.no_split,
        )
        model = load_checkpoint_and_dispatch(model, name_or_path, dmap)

    return tokenizer, model


def calc_f1_score(pred, target, average="all"):
    """Calculate the F1 score of the prediction"""
    assert len(pred) == len(target), f"Length mismatch: {len(pred)} != {len(target)}"
    if average == "all":
        return {
            a: f1_score(target, pred, average=a) for a in ["micro", "macro", "weighted"]
        }
    return f1_score(target, pred, average=average)


def load_dataset_splits(dataset, task, val_pct=20):
    builder = load_dataset_builder(dataset, task, trust_remote_code=True)
    for s in ("train", "test"):
        assert s in builder.info.splits, f"{dataset}/{task} doesn't contain a {s} set!"

    if "validation" in builder.info.splits:
        train_set = load_dataset(dataset, task, split="train", trust_remote_code=True)
        val_set = load_dataset(
            dataset, task, split="validation", trust_remote_code=True
        )
    else:  # split the train set as val set
        val_pct = 100 - val_pct
        train_set = load_dataset(
            dataset, task, split=f"train[:{val_pct}%]", trust_remote_code=True
        )
        val_set = load_dataset(
            dataset, task, split=f"train[{val_pct}%:]", trust_remote_code=True
        )
    test_set = load_dataset(dataset, task, split="test", trust_remote_code=True)

    def trec_rename(ds):
        if "coarse_label" in ds.features:
            return ds.rename_column("coarse_label", "label")
        return ds

    train_set, val_set, test_set = map(trec_rename, (test_set, val_set, test_set))
    return train_set, val_set, test_set


def convert_for_verbalizer(i2label, *support_and_query):
    def getter(x):
        for k in ["question", "sentence", "sentence1", "premise", "question1", "text"]:
            if k in x:
                return x[k]
        else:
            raise ValueError("No text found")

    samples = []
    for part in support_and_query:
        if part:
            for inp, out in zip(getter(part), part["label"]):
                samples.append({"input": inp, "output": i2label[out]})
    return samples


def verbalize(verbalizer, samples, device):
    full_token, ans_mask, io_mask = verbalizer(samples, io_sep_mask=True)
    full_token = {k: v.to(device) for k, v in full_token.items()}
    ans_mask = ans_mask.to(device)
    io_mask = io_mask.to(device)
    return full_token, ans_mask, io_mask


def get_i2label(vtask, train_set):
    if vtask == "sst5":
        labels = ["terrible", "negative", "neutral", "positive", "great"]
    elif vtask == "trec":
        labels = [
            "Abbreviation",
            "Entity",
            "Description",
            "Person",
            "Location",
            "Number",
        ]
    elif vtask == "emoji":
        labels = [
            "red_heart",
            "heart-eyes_smiling_face",
            "face_with_tears_of_joy",
            "two_hearts",
            "fire",
            "smiling_eyes_smiling_face",
            "sunglasses_smiling_face",
            "sparkles",
            "blue_heart",
            "face_blowing_a_kiss",
            "camera",
            "United_States",
            "sun",
            "purple_heart",
            "winking_face",
            "hundred_points",
            "beaming_face_with_smiling_eyes",
            "Christmas_tree",
            "camera_with_flash",
            "tongue_winking_face",
        ]
    else:
        return dict(enumerate(train_set.features["label"].names))

    return dict(zip(range(len(labels)), labels))


def last_only_io_mask(io_mask):
    ones_indices = io_mask.squeeze().nonzero()
    last_one_index = ones_indices[-1].item()
    last_only_mask = torch.zeros_like(io_mask)
    last_only_mask[..., last_one_index] = True
    return last_only_mask


class Verbalizer:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        example_sep="\n\n\n",
        input_output_sep="\n",
        max_len_per_example=256,
        instruction="",
    ):
        """Seperate the input and output with #`io_sep_cnt` `sep`s
        and #`ex_sep_cnt` `sep`s between examples"""
        self.tokenizer = tokenizer
        self.ex_sep_ids = tokenizer(example_sep, add_special_tokens=False)["input_ids"]
        self.io_sep_ids = tokenizer(input_output_sep, add_special_tokens=False)[
            "input_ids"
        ]
        self.max_len = max_len_per_example
        self.max_example_len = (
            max_len_per_example - len(self.ex_sep_ids) - len(self.io_sep_ids) + 1
        )
        self.ex_sep_mark = "<ex_sep>"
        self.io_sep_mark = "<io_sep>"
        self.inst_tokens = self.tokenizer.tokenize(instruction)
        tokenizer.add_special_tokens(
            {"sep_token": x for x in (self.ex_sep_mark, self.io_sep_mark)}
        )

    def format_one_sample(
        self, sample: dict, no_output=False, max_opt_len=0
    ) -> list[str]:
        """Format one sample into a list of tokenized strings

        `max_opt_len`: length to reserve for the longest option"""
        max_len = self.max_example_len - max_opt_len
        assert (
            max_len > 0
        ), f"max example length is too short, unable to fit option len {max_opt_len}"
        untrunc_text = f"{sample['input'].rstrip()}{self.io_sep_mark}{'' if no_output else sample['output']}"
        full_list = self.tokenizer.tokenize(untrunc_text)
        if len(full_list) <= max_len:
            return full_list

        # too long, truncate the input
        excess = len(full_list) - max_len
        io_sep_loc = full_list.index(self.io_sep_mark)
        trunc_list = full_list[: io_sep_loc - excess]
        trunc_list.append(self.io_sep_mark)  # keep the io_sep_mark
        if no_output:
            assert len(full_list) == io_sep_loc + 1
        else:
            trunc_list += full_list[io_sep_loc + 1 :]
        return trunc_list

    def get_token_and_mask(self, full_list):
        """Convert to tokens and find where the answer begins"""
        last_io_sep = len(full_list) - full_list[::-1].index(self.io_sep_mark) - 1
        quest_list = full_list[: last_io_sep + 1]
        full_token, add_len = self.convert_text_to_ids(full_list, return_tensors=True)
        ans_mask = torch.zeros_like(full_token["input_ids"], dtype=torch.bool)
        ans_mask[:, len(quest_list) + add_len :] = True
        return full_token, ans_mask

    def convert_text_to_ids(self, text: list[str], return_tensors=False):
        """Convert list of text with marks to ids
        and calculate additional length due to special tokens"""
        vocab = self.tokenizer.get_vocab()
        ret = []
        additional_len = 0
        for t in text:
            if t == self.ex_sep_mark:
                ret += self.ex_sep_ids
                additional_len += len(self.ex_sep_ids) - 1
            elif t == self.io_sep_mark:
                ret += self.io_sep_ids
                additional_len += len(self.io_sep_ids) - 1
            else:
                ret.append(vocab[t])
        if return_tensors:
            ret = {
                "input_ids": torch.tensor(ret, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones((1, len(ret)), dtype=torch.long),
            }
        return ret, additional_len

    def get_intermediate_ans_mask(self, text: list[str]):
        """Find where all the <io_sep> are and build a mask for them"""
        m = []
        for t in text:
            if t == self.io_sep_mark:
                m += [False] * (len(self.io_sep_ids) - 1)
                m.append(True)
            elif t == self.ex_sep_mark:
                m += [False] * len(self.ex_sep_ids)
            else:
                m.append(False)
        m = torch.tensor(m, dtype=torch.bool).unsqueeze(0)
        return m

    def __call__(self, samples: list, all_options=False, io_sep_mask=False):
        """Format and tokenize the given samples, return the tokenized text and the answer mask"""
        full_list = self.inst_tokens.copy()
        for s in samples[:-1]:
            full_list += self.format_one_sample(s)
            full_list.append(self.ex_sep_mark)

        if all_options:
            options = [(o, self.tokenizer.tokenize(o)) for o in samples[-1]["options"]]
            opt_to_lens = {o: len(tokens) for o, tokens in options}
            max_opt_len = max(opt_to_lens.values())
            full_list += self.format_one_sample(
                samples[-1], no_output=True, max_opt_len=max_opt_len
            )
            ret = []
            for _, tokens in options:
                an_option = full_list + tokens
                full_token, ans_mask = self.get_token_and_mask(an_option)
                if io_sep_mask:
                    io_mask = self.get_intermediate_ans_mask(an_option)
                    ret.append((full_token, ans_mask, io_mask))
                else:
                    ret.append((full_token, ans_mask))
            return ret
        else:
            full_list += self.format_one_sample(samples[-1])
            full_token, ans_mask = self.get_token_and_mask(full_list)
            if io_sep_mask:
                io_mask = self.get_intermediate_ans_mask(full_list)
                return full_token, ans_mask, io_mask
            return full_token, ans_mask


class FewshotDataset:
    def __init__(self, dataset, threshold=0.02):
        self.dataset = dataset
        self.threshold = threshold

        # exclude classes too rare to ensure enough examples
        ratio = torch.bincount(torch.tensor(dataset["label"])) / len(dataset)
        self.classes = [i for i, r in enumerate(ratio) if r > self.threshold]
        self.subsets = {
            i: dataset.filter(lambda x: x["label"] == i) for i in self.classes
        }

    def random_fewshot_support(self, rng, shot, shuffle=True):
        examples = []
        for subset in self.subsets.values():
            indices = rng.choice(len(subset), size=shot, replace=False)
            for i in indices:
                examples.append(subset[int(i)])
        if shuffle:
            rng.shuffle(examples)
        # merge into one dict
        support = {}
        for e in examples:
            for k, v in e.items():
                if k not in support:
                    support[k] = []
                support[k].append(v)
        return support

    @staticmethod
    def _check_duplicate(support, query):
        """Check if the query is in the support set. Return True if it is."""
        if not support or not query:
            return False
        for i in range(len(support["label"])):
            if not any([support[k][i] != query[k][0] for k in support.keys()]):
                return True
        return False

    def random_episode(self, rng, shot, shuffle_support=True):
        support = self.random_fewshot_support(rng, shot, shuffle=shuffle_support)
        query = random_examples(rng, self.dataset, 1)

        # check duplicate
        while self._check_duplicate(support, query):
            query = random_examples(rng, self.dataset, 1)

        return support, query
