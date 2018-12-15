import argparse
import csv
import logging
import os
import random
import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.training import train_and_evaluate

def import_names_and_labels(data_dir, name, num_labels, sample_size=None):
    path = os.path.join(data_dir, name + ".csv")
    pairs = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for i, row in enumerate(reader):
            name, label = row
            label = [int(i) for i in label.split()]
            label = [int(i in label) for i in range(num_labels)]
            pairs.append((name, label))
    if sample_size is None:
        random.shuffle(pairs)
    else:
        pairs = random.sample(pairs, sample_size)
    names, labels = zip(*pairs)
    return names, labels

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/00_base_model')
parser.add_argument('--data_dir', default='data/kaggle')
parser.add_argument('--restore_from', default=None)
parser.add_argument('--overwrite', default=False)
if __name__ == "__main__":
    random.seed(0)
    tf.set_random_seed(0)

    args = parser.parse_args()
    assert os.path.exists(args.model_dir)
    assert os.path.exists(args.data_dir)

    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)
    params.evaluate()

    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    if args.overwrite:
        for root, _, files in os.walk(args.model_dir):
            for f in files:
                if f != "params.json":
                    os.remove(os.path.join(root, f))
        for root, dirs, _ in os.walk(args.model_dir):
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    else:
        assert not model_dir_has_best_weights or args.restore_from is not None
    
    set_logger(os.path.join(args.model_dir, 'train.log'))
    

    logging.info('Creating the datasets...')
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    train_names, train_labels = import_names_and_labels(data_dir, "train", params.num_labels, params.train_size)
    eval_names, eval_labels = import_names_and_labels(data_dir, "dev", params.num_labels, params.dev_size)
    params.train_size = len(train_names)
    params.eval_size = len(eval_names)
    
    train_inputs = input_fn("train", images_dir, train_names, train_labels, params)
    eval_inputs = input_fn("eval", images_dir, eval_names, eval_labels, params)

    logging.info('Creating the model...')
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
    
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
