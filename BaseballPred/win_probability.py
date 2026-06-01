from torch import nn, optim
import torch
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import os
import random
import math
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from safetensors.torch import save_file
from .write_model_info import to_format_file
from cod_ml_core.win_predictor import WinPredictor

#python3 -m BaseballPred.win_probability

class BaseballData(IterableDataset):
    def __init__(self, file_list, scaler):
        super().__init__()

        self.files = file_list

        self.scaler = scaler

    def __iter__(self):
        #globally shuffle to emulate the shuffling of a standard tensordataset
        random.shuffle(self.files)

        worker_info = get_worker_info()
        if worker_info is None:
            worker_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / worker_info.num_workers))

            worker_id = worker_info.id

            start_idx = worker_id * per_worker
            end_idx = min((worker_id + 1) * per_worker, len(self.files))
            worker_files = self.files[start_idx:end_idx]


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
    
def create_global_scaler(data_files):
    scaler = StandardScaler()
    feature_cols = ["balls", "inning_topbot", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "run_diff"]

    for f in data_files:
        df = pd.read_parquet(f)
        chunk = df[feature_cols].astype(float)
        scaler.partial_fit(chunk)
    
    #this saves the scaler so that we don't have to recompute. If data changes, delete this
    joblib.dump(scaler, "baseball_scaler.pkl")
    return scaler
    
if __name__ == "__main__":

    train_files = []
    validate_files = []
    test_files = []

    i = 0
    for root, dirs, files in os.walk("./dataset"):
        #sort by year to avoid temporal leakeage
        for f in files:
            if "year=2024" in root:
                validate_files.append(root + "/" + f)
            elif "year=2025" in root:
                test_files.append(root + "/" + f)
            else:
                train_files.append(root + "/" + f)
        i += 1

    scaler_path = "basescaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = create_global_scaler(train_files)

    train_dataset = BaseballData(train_files, scaler)
    validate_dataset = BaseballData(validate_files, scaler)
    test_dataset = BaseballData(test_files, scaler)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        num_workers=4
    )

    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=512,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=512,
        num_workers=2
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = WinPredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("test")
    to_format_file(model, "model_info.txt")

    # epochs = 10
    # best_roc_auc = 0.0

    # for epoch in range(epochs):
    #     #just puts it in training mode
    #     model.train()
    #     running_loss = 0.0
        
    #     train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

    #     for batch_idx, (features, labels) in enumerate(train_pbar):
    #         features, labels = features.to(device), labels.to(device)
    #         #clears revious gradients
    #         optimizer.zero_grad()

    #         #forward pass
    #         predictions = model(features)

    #         #calculate how wrong forward pass was
    #         loss = criterion(predictions, labels)

    #         #backprop - i.e. update the gradients
    #         loss.backward()

    #         #update paremters using gradients according to Adam
    #         optimizer.step()

    #         running_loss += loss.item()

    #         train_pbar.set_postfix({"Loss": f"{running_loss/(batch_idx+1):.4f}"})


    #     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/(batch_idx+1):.4f}")

    #     model.eval()
    #     val_loss = 0.0

    #     all_targets = []
    #     all_predictions = []
    #     all_probabilities = []

    #     val_pbar = tqdm(validate_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

    #     #no need to track graidents during validation
    #     with torch.no_grad():
    #         for batch_idx, (val_features, val_labels) in enumerate(val_pbar):
    #             val_features, val_labels = val_features.to(device), val_labels.to(device)
    #             val_preds = model(val_features)
    #             loss = criterion(val_preds, val_labels)
    #             val_loss += loss.item()

    #             #convert logits to probabilites for analysis
    #             probabilities = F.softmax(val_preds, dim=1)[:, 1]

    #             #convert probs to hard predictions
    #             predicted_classes = torch.argmax(val_preds, dim=1)

    #             all_targets.extend(val_labels.cpu().numpy())
    #             all_probabilities.extend(probabilities.cpu().numpy())
    #             all_predictions.extend(predicted_classes.cpu().numpy())

    #             val_pbar.set_postfix({"Val Loss": f"{val_loss/(batch_idx+1):.4f}"})

    #     avg_val_loss = val_loss / (batch_idx + 1)
    #     accuracy = accuracy_score(all_targets, all_predictions)
    #     precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary', zero_division=0)
    #     roc_auc = roc_auc_score(all_targets, all_probabilities)
    #     print(f"\n--- Epoch {epoch+1} Validation ---")
    #     print(f"Loss:      {avg_val_loss:.4f}")
    #     print(f"Accuracy:  {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    #     print(f"ROC-AUC:   {roc_auc:.4f}\n")

    #     if roc_auc > best_roc_auc:
    #         best_roc_auc = roc_auc
    #         print(f"New best model found on Epoch {epoch + 1}")
    #         save_file(model.state_dict(), "best_baseball_predictor.safetensors")