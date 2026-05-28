import grpc
import model_pb2
import model_pb2_grpc

#command to compile proto: python3 -m grpc_tools.protoc -I ../rust_grpc_server/proto --python_out=. --grpc_python_out=. ../rust_grpc_server/proto/model.proto

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = model_pb2_grpc.WeightsManagerStub(channel)

        gradients_list = [0.01, -0.05, 0.12, 0.05]
        request = model_pb2.WeightRequest(gradients=gradients_list)

        print("Sending gradients to Rust Server...")
        response = stub.UpdateWeights(request)

        print(f"Rust server responded. Success: {response.success}")

if __name__ == '__main__':
    run()