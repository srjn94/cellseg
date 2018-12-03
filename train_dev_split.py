import argparse
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kaggle')
parser.add_argument('--ratio', nargs=3, default=[98,1,1], metavar=('train','dev','test'), type=float)

if __name__ == "__main__":
    random.seed(1)
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.data_dir, 'index.csv'))
    df_train, df_devtest = train_test_split(df, test_size=sum(args.ratio[1:])/sum(args.ratio))
    df_dev, df_test = train_test_split(df_devtest, test_size=args.ratio[2]/sum(args.ratio[1:]))
    df_train.sort_index(inplace=True)
    df_dev.sort_index(inplace=True)
    df_test.sort_index(inplace=True)
    df_train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    df_dev.to_csv(os.path.join(args.data_dir, 'dev.csv'), index=False)
    df_test.to_csv(os.path.join(args.data_dir, 'test.csv'), index=False)
