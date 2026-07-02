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
        
        print(worker_reg.success)
        print(worker_reg.worker_id)
        print(worker_reg)

        if worker_reg.success:
            worker_id = worker_reg.worker_id
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


                

if __name__ == '__main__':
    run()