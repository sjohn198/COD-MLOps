use tonic::{transport::Server, Request, Response, Status};
use parameter_server::weights_manager_server::{WeightsManager, WeightsManagerServer};
use parameter_server::{WeightResponse, WeightsRequest, LayerGradient, ModelUpdate, GoodMorning, WorkerRegistration};
use memmap2::MmapOptions;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::fs::File;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};


pub mod parameter_server {
    tonic::include_proto!("parameter_server");
}

fn read_weights() -> Result<BTreeMap<String, Vec<f32>>, Box<dyn std::error::Error>> {
    let file = File::open("../../BaseballPred/best_baseball_predictor.safetensors")?;
    let buffer = unsafe { MmapOptions::new().map(&file)?};
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut weights_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();

    for (layer_name, tensor_view) in tensors.tensors() {
        println!("Layer: {:?}, Shape: {:?}", layer_name, tensor_view.shape());
        let weight_array = extract_f32_vector(&tensor_view)?;
        weights_map.insert(layer_name.to_string(), weight_array);
    }
    Ok(weights_map)
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
pub struct BaseballWeightsManager {
    pub weights: Arc<Mutex<BTreeMap<String, Vec<f32>>>>,
    pub num_conns: Arc<AtomicI32>
}

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
    async fn request_weights(
        &self,
        request: Request<WeightsRequest>
    ) -> Result<Response<ModelUpdate>, Status> {
        let req_data = request.into_inner();
        println!("Worker id: {}", req_data.worker_id);
        
        let state = self.weights.lock().await;
        let mut response_layers = Vec::with_capacity(state.len());

        for (i, (layer_name, weights)) in state.iter().enumerate() {
            let layer_id = i as i32;

            response_layers.push(LayerGradient {
                id: layer_id,
                weights: weights.clone()
            })
        }

        let reply = ModelUpdate { layers: response_layers };
        Ok(Response::new(reply))
    }
    async fn wake_worker(
        &self,
        request: Request<GoodMorning>
    ) -> Result<Response<WorkerRegistration>, Status> {
        let prev_num_conn = self.num_conns.fetch_add(1, Ordering::SeqCst);

        let worker_id = prev_num_conn; //may want to make this pnc + 1
        let reply = WorkerRegistration {
            worker_id: worker_id
        };
        Ok(Response::new(reply))
        //TODO: will also need a worker deregistration sort of thing once a work is done w all training
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weights = read_weights()?;

    let addr = "[::1]:50051".parse()?;
    let manager = BaseballWeightsManager {
        weights: Arc::new(Mutex::new(weights)),
        num_conns: Arc::new(AtomicI32::new(0))
    };

    println!("gRPC Server listening on {}", addr);

    Server::builder()
        .add_service(WeightsManagerServer::new(manager))
        .serve(addr)
        .await?;

    Ok(())
}