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

#command to compile proto: python3 -m grpc_tools.protoc -I ../rust_grpc_server/proto --python_out=. --grpc_python_out=. ../rust_grpc_server/proto/model.proto

def run():
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

        layer_names = sorted(state_dict.keys())

        with torch.no_grad():
            for layer_data in layers_resp.layers:
                layer_name = layer_names[layer_data.id]
                target_shape = state_dict[layer_name].shape
                flat_tensor = torch.tensor(layer_data.weights, dtype=torch.float32)
                reshaped_tensor = flat_tensor.view(target_shape)
                state_dict[layer_name].copy_(reshaped_tensor)

            model.load_state_dict(state_dict)
            print("Model weights successfully fully synchronized with Rust server!")

            updated_sd = model.state_dict()
            layers = []
            for layer_id, layer_name in enumerate(layer_names):
                weights = updated_sd[layer_name]
                w = weights.flatten().tolist()
                layer = model_pb2.LayerGradient(id=layer_id, weights=w)
                layers.append(layer)

            request = model_pb2.ModelUpdate(layers=layers)
            response = stub.UpdateWeights(request)

            print(f"Rust server responded. Success: {response.success}")

            scaler = joblib.load("./data/baseball_scaler.pkl")

            print("scaler loaded properly")

            try:
                with open("./data/train_files.json", "r") as f1:
                    train_files = list(json.load(f1))
                with open("./data/test_files.json", "r") as f2:
                    test_files = list(json.load(f2))
                with open("./data/validate_files.json", "r") as f3:
                    validate_files = list(json.load(f3))
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

if __name__ == '__main__':
    run()