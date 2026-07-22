import grpc
import model_pb2
import model_pb2_grpc
from win_predictor import WinPredictor
import torch
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from baseball_iterable_dataset import BaseballData
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import sys
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, accuracy_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve, CalibrationDisplay
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import save_file

#command to compile proto: python3 -m grpc_tools.protoc -I ../rust_grpc_server/proto --python_out=. --grpc_python_out=. ../rust_grpc_server/proto/model.proto

def verify_data_files():
    filepaths = [
        "./data/train_files.json",
        "./data/test_files.json",
        "./data/validate_files.json"
    ]

    for f in filepaths:
        if not os.path.exists(f):
            print(f"Missing file: {f}")
            sys.exit(1)

    with open("./data/train_files.json", "r") as f1:
        train_files = json.load(f1)
        if len(train_files) == 0:
            print("train_files.json exists but is empty")
            sys.exit(1)
    
    with open("./data/test_files.json", "r") as f2:
        test_files = json.load(f2)
        if len(test_files) == 0:
            print("test_files.json exists but is empty")
            sys.exit(1)

    with open("./data/validate_files.json", "r") as f3:
        validate_files = json.load(f3)
        if len(validate_files) == 0:
            print("validate_files.json exists but is empty")
            sys.exit(1)

    try:
        scaler = joblib.load("./data/baseball_scaler.pkl")

        if not hasattr(scaler, 'mean_'):
            print("Scaler was saved but never fitted")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to load scaler: {e}")
        sys.exit(1)


def run():
    verify_data_files()
    with grpc.insecure_channel(os.environ.get("GRPC_SERVER_ADDRESS", "localhost:50051")) as channel:
        stub = model_pb2_grpc.WeightsManagerStub(channel)

        init_request = model_pb2.GoodMorning()

        worker_reg = stub.WakeWorker(init_request)

        worker_id = None
        path = None
        
        print(worker_reg.success)
        print(worker_reg.worker_id)
        print(worker_reg)

        if worker_reg.success:
            worker_id = worker_reg.worker_id
            path = worker_reg.path
        else:
            print("too many workers assigned")
            return

        print(f"Assigned Worker ID: {worker_id}")

        model_request = model_pb2.WeightsRequest(worker_id=worker_id)

        layers_resp = stub.RequestWeights(model_request)

        model = WinPredictor()
        state_dict = model.state_dict()

        layer_names = list(state_dict.keys())
        for layer_data in layers_resp.layers:
            layer_name = layer_names[layer_data.id]
            target_shape = state_dict[layer_name].shape
            flat_tensor = torch.tensor(layer_data.weights, dtype=torch.float32)
            reshaped_tensor = flat_tensor.view(target_shape)
            state_dict[layer_name].copy_(reshaped_tensor)

        model.load_state_dict(state_dict)
        print("Model weights successfully synchronized with Rust server!")

        updated_sd = model.state_dict()
        # layers = []
        # for layer_id, layer_name in enumerate(layer_names):
        #     weights = updated_sd[layer_name]
        #     w = weights.flatten().tolist()
        #     layer = model_pb2.LayerGradient(id=layer_id, weights=w)
        #     layers.append(layer)

        # request = model_pb2.ModelUpdate(layers=layers)
        # response = stub.UpdateWeights(request)

        # print(f"Rust server responded. Success: {response.success}")

        scaler = joblib.load("./data/baseball_scaler.pkl")

        print("scaler loaded properly")

        try:
            with open("./data/train_files.json", "r") as f1:
                train_files = list(json.load(f1))
                print(f"Loaded {len(train_files)} files.")
            with open("./data/test_files.json", "r") as f2:
                test_files = list(json.load(f2))
                print(f"Loaded {len(test_files)} files.")
            with open("./data/validate_files.json", "r") as f3:
                validate_files = list(json.load(f3))
                print(f"Loaded {len(validate_files)} files.")
        except FileNotFoundError:
            print("data files not intact")
            return
        
        train_dataset = BaseballData(train_files, scaler, worker_id, 4)
        test_dataset = BaseballData(test_files, scaler, worker_id, 4)
        validate_dataset = BaseballData(validate_files, scaler, worker_id, 4)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=512
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=512
        )

        validate_loader = DataLoader(
            dataset=validate_dataset,
            batch_size=512
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model.to(device)

        #gotta calculate adam and loss in the rust file

        
        print(f"Device selected: {device}")

        criterion = nn.CrossEntropyLoss()
        epochs = 10
        best_brier = 1
        briers = []
        #print("Registered Parameters:", list(model.named_parameters()))

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            print(f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, (features, labels) in enumerate(train_pbar):
                features, labels = features.to(device), labels.to(device)

                #zero out optimizer from server

                predictions = model(features)
                loss = criterion(predictions, labels)
                loss.backward()

                layers = []

                for layer_id, (layer_name, param) in enumerate(model.named_parameters()):
                    if param.requires_grad and param.grad is not None:
                        grad_list = param.grad.flatten().tolist()

                        layer = model_pb2.LayerGradient(id=layer_id, weights=grad_list)
                        layers.append(layer)

                #print("Model update")
                model_update = model_pb2.ModelUpdate(layers=layers)
                send_update = stub.UpdateWeights(model_update)

                weight_request = model_pb2.WeightsRequest(worker_id=worker_id)
                layers_resp = stub.RequestWeights(weight_request)

                current_state = model.state_dict()
                keys_list = list(current_state.keys())

                for layer in layers_resp.layers:
                    name = keys_list[layer.id]
                    shape = current_state[name].shape

                    flat_tensor = torch.tensor(layer.weights, dtype=torch.float32)
                    current_state[name].copy_(flat_tensor.view(shape))

                model.load_state_dict(current_state)
                model.zero_grad()

                running_loss += loss.item()

                train_pbar.set_postfix({"Loss": f"{running_loss/(batch_idx+1):.4f}"})
            
            if worker_id == 0:
                weight_request = model_pb2.WeightsRequest(worker_id=worker_id)
                layers_resp = stub.RequestWeights(weight_request)

                current_state = model.state_dict()
                keys_list = list(current_state.keys())

                for layer in layers_resp.layers:
                    name = keys_list[layer.id]
                    shape = current_state[name].shape

                    flat_tensor = torch.tensor(layer.weights, dtype=torch.float32)
                    current_state[name].copy_(flat_tensor.view(shape))

                model.load_state_dict(current_state)

                model.eval()
                val_loss = 0.0

                all_targets = []
                all_predictions = []
                all_probabilities = []

                val_pbar = tqdm(validate_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
                with torch.no_grad():
                    for batch_idx, (val_features, val_labels) in enumerate(val_pbar):
                        val_features, val_labels = val_features.to(device), val_labels.to(device)
                        val_preds = model(val_features)
                        loss = criterion(val_preds, val_labels)
                        val_loss += loss.item()

                        probabilities = F.softmax(val_preds, dim=1)[:, 1]
                        predicted_classes = torch.argmax(val_preds, dim=1)

                        #pulling off the gpu is super inefficient
                        all_targets.extend(val_labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())
                        all_predictions.extend(predicted_classes.cpu().numpy())

                        val_pbar.set_postfix({"Val Loss": f"{val_loss/(batch_idx+1):.4f}"})                        
                    
                    final_targets = np.array(all_targets)
                    final_probs = np.array(all_probabilities)

                    avg_val_loss = val_loss / (batch_idx+1)
                    brier = brier_score_loss(all_targets, all_probabilities)
                    acc = accuracy_score(all_targets, all_predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary', zero_division=0)
                    print(f"\n--- Epoch {epoch+1} Validation ---")
                    print(f"Loss:      {avg_val_loss:.4f}")
                    print(f"Brier:     {brier}")
                    print(f"Accuracy:  {acc:.4f}")
                    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

                    briers.append(brier)
                    if brier < best_brier:
                        best_brier = brier
                        print(f"New best model found on epoch {epoch + 1}")
                        save_file(model.state_dict(), f"{path}/best_baseball_predictor.safetensors")


                    fig, ax = plt.subplots(figsize=(8,8))
                    display = CalibrationDisplay.from_predictions(
                        y_true=final_targets,
                        y_prob=final_probs,
                        n_bins=10,
                        name="Win Predictor Epoch{}".format(epoch+1),
                        ax=ax,
                        strategy="uniform"
                    )
                    
                    ax.set_title(f"Calibration Curve - Epoch {epoch+1}")
                    ax.grid(True, linestyle='--', alpha=0.7)

                    # Save the image to the /app/data directory so it appears on your host Mac
                    plt.savefig(f"{path}/calibration_epoch_{epoch+1}.png", dpi=300, bbox_inches='tight')

                    # Close the figure to free up memory
                    plt.close(fig)
        if worker_id == 0:
            epoch_numbers = list(range(1, epochs + 1))
            fig2, ax2 = plt.subplots(figsize=(10,6))

            ax2.plot(epoch_numbers, briers, marker='o', linestyle='-', color='blue', markersize=6)

            for i, score in enumerate(briers):
                ax2.annotate(
                    f'{score}',
                    (epoch_numbers[i], score),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9
                )   
            ax.set_title("Validation Brier Score per Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Brier Score (Lower is Better)")
            ax.set_xticks(epoch_numbers) # Forces the X-axis to only show whole epoch numbers
            ax.grid(True, linestyle='--', alpha=0.6)

            plt.savefig(f"{path}/brier_score_history.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)

if __name__ == '__main__':
    run()