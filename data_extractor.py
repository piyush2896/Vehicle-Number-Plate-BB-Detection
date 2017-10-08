import numpy as np
import pandas as pd


def extract_data(csv):
    df = pd.read_csv(csv)
    x = df['pixels'].apply(lambda im: np.fromstring(im, sep=', '))
    x = np.vstack(x.values)
    x= x.astype(np.float32)
    y = np.array(df[['x1', 'y1',
            'x2', 'y2',
            'x3', 'y3',
            'x4', 'y4',
           ]])
    return x, y


def batches_(x, y, batch_size=24):
    batches = np.arange(x.shape[0])
    np.random.shuffle(batches)
    if x.shape[0]%batch_size == 0:
        batches = batches.reshape(x.shape[0]//batch_size, -1)
    else:
        values = np.arange(batch_size - x.shape[0] % batch_size)
        np.random.shuffle(values)
        batches = np.append(batches, values)
        batches = batches.reshape(x.shape[0]//batch_size + 1, -1)
    for i in range(batches.shape[0]):
        yield x[batches[i]], y[batches[i]]

