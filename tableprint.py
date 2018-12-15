import json
import os
from collections import defaultdict

root = "experiments"
dirs = sorted(os.listdir(root))
dirswidth = max(map(len, dirs))

print(f'{"experiment": <{1 + dirswidth}}\t loss\t f1_score')
for d in dirs:
    try:
        with open(os.path.join(root, d, "metrics_eval_best_weights.json"), "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = defaultdict(lambda: float("NaN"))
    print(f'{d: <{dirswidth}}\t{data["loss"]: .4f}\t{data["macro_f1_score"]: .4f}')
    


