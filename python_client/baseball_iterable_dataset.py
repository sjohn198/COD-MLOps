import torch
import random
import math
import pandas as pd
from torch.utils.data import IterableDataset

class BaseballData(IterableDataset):
    def __init__(self, file_list, scaler, worker_id, num_workers):
        super().__init__()

        self.files = file_list

        self.scaler = scaler

        self.worker = worker_id

        self.num_workers = num_workers

    def __iter__(self):
        #globally shuffle to emulate the shuffling of a standard tensordataset
        self.files.sort()

        per_worker = int(math.ceil(len(self.files) / self.num_workers))

        start_idx = self.worker * per_worker
        end_idx = min((self.worker + 1) * per_worker, len(self.files))
        worker_files = self.files[start_idx:end_idx]

        random.shuffle(self.files)

        for f in worker_files:
            df = pd.read_parquet(f)

            df = df.sample(frac=1).reset_index(drop=True)

            x_dataset = df[["balls", "inning_topbot", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "run_diff"]].astype(float)
            y_dataset = df["winner"]

            x_scaled = self.scaler.transform(x_dataset)

            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y_dataset.values, dtype=torch.long)

            for i in range(len(x_tensor)):
                yield x_tensor[i], y_tensor[i]
    