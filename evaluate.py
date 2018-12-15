import argparse
import csv
import logging
import numpy as np
import os
import random
import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger
from train import import_names_and_labels

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments/07_full_images")
parser.add_argument("--data_dir", default="data/kaggle")
parser.add_argument("--restore_from", default="best_weights")
parser.add_argument("--set", default="test")

if __name__ == "__main__":
    tf.set_random_seed(230)
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    params = Params(json_path)
    params.evaluate()

    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    image_dir = os.path.join(data_dir, "images")
    names, labels = import_names_and_labels(data_dir, "test", params.num_labels)
    params.eval_size = len(names)
    inputs = input_fn("test", image_dir, names, labels, params)
    
    logging.info("Creating the model...")
    model_spec = model_fn("eval", inputs, params)

    logging.info("Evaluating...")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
