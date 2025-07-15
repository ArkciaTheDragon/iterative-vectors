# This script file is written for Fish shell.
# You may need to adapt it if you are using a different shell, e.g. bash.

# Optional environment variables

set -x PYTHONWARNINGS ignore::DeprecationWarning  # a baukit issue
set -x DATASETS_VERBOSITY error
set -x TRANSFORMERS_VERBOSITY error
set -x HF_DATASETS_OFFLINE 1  # set this when all datasets are cached

# Iterative Vectors (Ours)

set gpu <your_gpu_id>
set config <your_config_name>
set model llama-2-7b
set episodes 200
set start_time (date +"%Y-%m-%d %H:%M:%S")
for t in $tasks
set task (get_task $t)
for ext_shot in 1 2 3 4
for strength in .1 .3 .5 .7 .9
for ext_strength in .1 .3 .5 .7 .9
echo $config EDIT on $task \| model=$model ext_shot=$ext_shot episodes=$episodes strength=$strength ext_strength=$ext_strength
python iv.py -c config/$config.yaml -g $gpu -m $model -t $task -n $episodes -s $ext_shot --strength $strength --ext-strength $ext_strength
echo
end
end
end
end
echo START TIME $start_time \| END TIME (date +"%Y-%m-%d %H:%M:%S")

# Automated Test
#
# You can use param_picker.ipynb to find the best hyperparameters and set them

set gpu <your_gpu_id>
set config <your_config_name>
for i in (seq (count $tasks))
set task $tasks[$i]
set strength $strengths[$i]
set ext_strength $ext_strengths[$i]
set ext_shot $ext_shots[$i]
echo $config TEST on $task \| model=$model ext_shot=$ext_shot strength=$strength ext_strength=$ext_strength
python iv.py -c config/$config.yaml -m $model -t $task -g $gpu -s $ext_shot -e $episodes --strength $strength --ext-strength $ext_strength --run-test
echo
end
echo TEST $model on $tasks
echo START TIME $start_time \| END TIME (date +"%Y-%m-%d %H:%M:%S")

# Function Vectors

set config <your_config_name>
set models llama-2-7b llama-2-13b gpt-j-6b llama-3.1-8b
set n_top_heads 20 50 10 25
set start_time (date +"%Y-%m-%d %H:%M:%S")
for i in (seq (count $models))
set model $models[$i]
set heads $n_top_heads[$i]
for task in $tasks
for ext_shot in 1
echo $config FV on $task \| model=$model ext_shot=$ext_shot n_top_heads=$heads
python fv.py -c config/$config.yaml -g $gpu -m $model -t $task -s $ext_shot --n-top-heads $heads
echo
end
end
end
echo FV $models: $tasks
echo START TIME $start_time \| END TIME (date +"%Y-%m-%d %H:%M:%S")

# Task Vectors

set model llama-2-7b
set start_time (date +"%Y-%m-%d %H:%M:%S")
for task in $tasks
for ext_shot in 1 2 3 4
echo $config TV on $task \| model=$model ext_shot=$ext_shot seed=$seed
python tv.py -c config/$config.yaml -g $gpu -m $model -t $task -s $ext_shot
echo
end
end
echo START TIME $start_time
echo END TIME (date +"%Y-%m-%d %H:%M:%S")
