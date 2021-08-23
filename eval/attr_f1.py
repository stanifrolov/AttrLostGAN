import argparse

import numpy as np
import pandas as pd
import sklearn.metrics as metrics


# Helper method to get micro F1 score
def get_score(filename, th=0.5, avg='micro'):
    file = pd.read_csv(filename, header=0)
    preds = file['prediction']
    targs = file['target']

    p = preds.replace('\n', ' ')
    t = targs.replace('\n', ' ')

    p_arr = []
    t_arr = []
    for i in range(len(p)):
        p_arr.append(np.asarray(p[i][1:-1].split()[0:], dtype=np.float32))
        t_arr.append(np.asarray(t[i][1:-1].split()[0:], dtype=np.float32))

    predictions = np.vstack(p_arr)
    targets = np.vstack(t_arr)

    f1_micro = metrics.f1_score(targets > 0.5, predictions > th, average=avg)

    return f1_micro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None,
                        help='Path to test predictions.csv at lowest validation loss')
    args = parser.parse_args()

    score = get_score(args.path)
    print(score)
