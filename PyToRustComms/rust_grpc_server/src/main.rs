use tonic::{transport::Server, Request, Response, Status};
use parameter_server::weights_manager_server::{WeightsManager, WeightsManagerServer};
use parameter_server::{WeightResponse, LayerGradient, ModelUpdate};
use memmap2::MmapOptions;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::fs::File;

pub mod parameter_server {
    tonic::include_proto!("parameter_server");
}

fn read_weights() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("../../BaseballPred/best_baseball_predictor.safetensors")?;
    let buffer = unsafe { MmapOptions::new().map(&file)?};
    let tensors = SafeTensors::deserialize(&buffer)?;

    // let sorted_layers: Vec<_> = tensors.tensors().collect();

    for (layer_name, tensor_view) in tensors.tensors() {
        println!("Layer: {:?}, Shape: {:?}", layer_name, tensor_view.shape());
        let weight_array = extract_f32_vector(&tensor_view)?;
    }
    Ok(())
}

fn extract_f32_vector(view: &TensorView) -> Result<Vec<f32>, String> {
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(format!("Expected F32, got {:?}", view.dtype()));
    }

    let raw_bytes: &[u8] = view.data();
    let mut float_array: Vec<f32> = Vec::with_capacity(raw_bytes.len() / 4);

    for chunk in raw_bytes.chunks_exact(4) {
        let val: f32 = f32::from_le_bytes(chunk.try_into().unwrap());
        float_array.push(val);
    }
    Ok(float_array)
}

#[derive(Debug, Default)]
pub struct BaseballWeightsManager {} //likely put global model weights in here wrapped in a mutex 

#[tonic::async_trait]
impl WeightsManager for BaseballWeightsManager {
    async fn update_weights(
        &self,
        request: Request<ModelUpdate>
    ) -> Result<Response<WeightResponse>, Status> {
        let req_data = request.into_inner();
        for layer in req_data.layers {
            println!("Layer id: {}", layer.id);
            println!("Receiced {} gradients.", layer.weights.len());
        }

        let reply = WeightResponse { success: true };
        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    read_weights();
    let addr = "[::1]:50051".parse()?;
    let manager = BaseballWeightsManager::default();

    println!("gRPC Server listening on {}", addr);

    Server::builder()
        .add_service(WeightsManagerServer::new(manager))
        .serve(addr)
        .await?;

    Ok(())
}