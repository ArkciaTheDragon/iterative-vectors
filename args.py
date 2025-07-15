import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

from yaml import safe_load


def merge_config(base, config):
    for k, v in config.items():
        if isinstance(v, dict):
            base[k] = merge_config(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def parse_args(argv=None, base_config_fn="config/base.yaml"):
    p = ArgumentParser()
    # yapf: disable
    p.add_argument("--config-base", type=Path, help="Base config file", default=base_config_fn)
    p.add_argument("-c", "--config", type=Path, default=None, help="Path to config file")

    p.add_argument("-t", "--task", type=str, help="Dataset and task name")
    p.add_argument("-m", "--model", type=str, required=True, help="Model name or path to model directory")
    p.add_argument("-g", "--gpu", nargs="*", type=int,
                   help="GPU device ids; Accelerate mapping will be used if more than one GPUs are provided")
    p.add_argument("--mem-limit", type=int, default=0, help="Max memory on the first GPU in GiB; 0 to disable")
    p.add_argument("--no-half", action="store_true", help="Don't use half precision")
    p.add_argument("--device-map", default=None,
                   help="Device mapping for multi-GPU training, e.g., balanced")

    p.add_argument("-e", "--episodes", type=int, default=200, help="Number of extraction episodes")
    p.add_argument("-s", "--ext-shot", type=int, default=1, help="Number of shots for IV extraction")
    p.add_argument("--strength", type=float, default=1, help="Strength of IV edit")
    p.add_argument("--ext-strength", type=float, default=.1, help="Strength of IV extraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--ext-batch", type=int, default=10, help="Extraction batch size; 0 to disable")
    
    p.add_argument("--shot", type=int, default=1, help="Few-shot evaluation shot")
    p.add_argument("-n", "--eval-episodes", type=int, default=10000, help="Number of evaluation episodes")
    p.add_argument("--run-clean", action="store_true", help="Don't use IV")
    p.add_argument("--run-test", action="store_true", help="Use the test split")
    p.add_argument("--flash-attention", action="store_true", help="Use flash attention 2")

    # FV related
    p.add_argument("--n-top-heads", type=int, default=20, help="Number of top heads to calc FV")
    p.add_argument("--ie-episodes", type=int, default=200,
                   help="Number of episodes to extract indirect effects")
    # yapf: enable

    # Load config
    if argv is None:
        args = p.parse_args()
    else:
        args = p.parse_args(argv)
    if not args.config_base.is_file():
        p.error(f"Base config file {args.config} not found")
    with args.config_base.open() as f:
        config = safe_load(f)
    if isinstance(args.config, Path) and args.config.is_file():
        with args.config.open() as f:
            config = merge_config(config, safe_load(f))
    else:
        warnings.warn(f"Config file {args.config} not found, using base config only")

    # Model and GPU checks
    if args.model not in config["models"]:
        p.error(f"Model {args.model} not found in config")
    if len(args.gpu) > 1:
        if args.device_map is None:
            p.error("--device-map is required when using multiple GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))

    # Populate target layers
    args.no_split = config["no_split_layer"][args.model]
    layer_name, start, end = config["target_layer"][args.model]
    args.target_layers = [layer_name.format(i) for i in range(start, end)]
    layer_name, start, end = config["tv_target_layer"][args.model]
    args.tv_target_layers = [layer_name.format(i) for i in range(start, end)]

    # Merge config
    args.model_name = args.model
    args.model = config["models"][args.model]
    for k, v in config["merge"].items():
        setattr(args, k, v)

    # Create result directories
    for fn in ("clean", "edit", "test"):
        if f := getattr(args, f"{fn}_result_fn", None):
            p = Path(f).parent
            p.mkdir(parents=True, exist_ok=True)

    # Determine task name
    if args.task:
        args.dataset, _, args.task = args.task.partition("/")
        args.vtask = args.task or args.dataset
        if args.dataset == "SetFit":  # sst5
            args.dataset = f"{args.dataset}/{args.task}"
            args.task = ""

    return args
