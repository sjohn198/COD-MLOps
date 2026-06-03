import grpc
import model_pb2
import model_pb2_grpc
from win_predictor import WinPredictor
import torch
import os

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
        

        # gradients_list = [0.01, -0.05, 0.12, 0.05]
        # l1 = model_pb2.LayerGradient(id=0, weights=gradients_list)
        # gradients_list2 = [0.02, -0.07, 0.21, 0.08]
        # l2 = model_pb2.LayerGradient(id=1, weights=gradients_list2)
        # layers = [l1, l2]
        # request = model_pb2.ModelUpdate(layers=layers)

        # print("Sending gradients to Rust Server...")
        # response = stub.UpdateWeights(request)

        # print(f"Rust server responded. Success: {response.success}")

if __name__ == '__main__':
    run()