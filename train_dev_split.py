import argparse
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kaggle')
parser.add_argument('--ratio', nargs=2, default=[99,1], metavar=('train','dev'), type=float)

if __name__ == "__main__":
    random.seed(1)
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.data_dir, 'index.csv'))
    df_train, df_dev = train_test_split(df, test_size=args.ratio[1]/sum(args.ratio))
    df_train.sort_index(inplace=True)
    df_dev.sort_index(inplace=True)
    df_train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    df_dev.to_csv(os.path.join(args.data_dir, 'dev.csv'), index=False)
