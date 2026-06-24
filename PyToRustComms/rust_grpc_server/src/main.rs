use tonic::{transport::Server, Request, Response, Status};
use parameter_server::weights_manager_server::{WeightsManager, WeightsManagerServer};
use parameter_server::{WeightResponse, WeightsRequest, LayerGradient, ModelUpdate, GoodMorning, WorkerRegistration};
//use memmap2::MmapOptions;
//use safetensors::tensor::TensorView;
//use safetensors::SafeTensors;
use std::fs::File;
use std::collections::{ HashMap };
use std::sync::Arc;
use tokio::sync::{ Mutex, MutexGuard };
use std::sync::atomic::{AtomicI32, Ordering};
use candle_core::{Device, Tensor, Result};
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::io::BufReader;


pub mod parameter_server {
    tonic::include_proto!("parameter_server");
}

/*we only need this if we're finetuning - which for now we're not */
// fn read_weights() -> std::result::Result<BTreeMap<String, Vec<f32>>, Box<dyn std::error::Error>> {
//     let file = File::open("/app/best_baseball_predictor.safetensors")
//                 .or_else(|_| File::open("best_baseball_predictor.safetensors"))?;
//     let buffer = unsafe { MmapOptions::new().map(&file)?};
//     let tensors = SafeTensors::deserialize(&buffer)?;

//     let mut weights_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();

//     for (layer_name, tensor_view) in tensors.tensors() {
//         println!("Layer: {:?}, Shape: {:?}", layer_name, tensor_view.shape());
//         let weight_array = extract_f32_vector(&tensor_view)?;
//         weights_map.insert(layer_name.to_string(), weight_array);
//     }
//     Ok(weights_map)
// }

// fn extract_f32_vector(view: &TensorView) -> std::result::Result<Vec<f32>, String> {
//     if view.dtype() != safetensors::Dtype::F32 {
//         return Err(format!("Expected F32, got {:?}", view.dtype()));
//     }

//     let raw_bytes: &[u8] = view.data();
//     let mut float_array: Vec<f32> = Vec::with_capacity(raw_bytes.len() / 4);

//     for chunk in raw_bytes.chunks_exact(4) {
//         let val: f32 = f32::from_le_bytes(chunk.try_into().unwrap());
//         float_array.push(val);
//     }
//     Ok(float_array)
// }

fn calculate_moments(
    first_moment: &Tensor, 
    second_moment: &Tensor, 
    first_decay_rate: f32,
    second_decay_rate: f32,
    gradients: &Tensor,
    iteration: i32) ->  Result<(Tensor, Tensor)>{
        let decayed_gradients: Tensor = gradients.affine((1.0 - first_decay_rate) as f64, 0.0)?;
        let decayed_first_moment: Tensor = first_moment.affine(first_decay_rate as f64, 0.0)?;
        let new_first_moment: Tensor = (decayed_first_moment + &decayed_gradients)?;

        let squared_gradients: Tensor = (gradients * gradients)?;
        let decayed_squared_gradients: Tensor = squared_gradients.affine((1.0 - second_decay_rate) as f64, 0.0)?;
        let decayed_second_moment: Tensor = second_moment.affine(second_decay_rate as f64, 0.0)?;
        let new_second_moment: Tensor = (decayed_second_moment + &decayed_squared_gradients)?;

        let bias_corrected_first_moment: Tensor = new_first_moment.affine((1.0 / (1.0 - first_decay_rate.powi(iteration))) as f64, 0.0)?;
        let bias_corrected_second_moment: Tensor = new_second_moment.affine((1.0 / (1.0 - second_decay_rate.powi(iteration))) as f64, 0.0)?;

        Ok((bias_corrected_first_moment, bias_corrected_second_moment))
}   


#[derive(Serialize, Deserialize, Default, Debug)]
#[serde(transparent)]
pub struct NetworkMap {
    pub layers: Vec<(i32, i32)>
}

fn read_config_from_file<P: AsRef<Path>>(path: P) -> std::result::Result<NetworkMap, Box<dyn std::error::Error>> {
    let file: File = File::open(path)?;
    let buffer: BufReader<File> = BufReader::new(file);
    let network_map: NetworkMap = serde_json::from_reader(buffer)?;

    Ok(network_map)
}

#[derive(Debug)]
pub struct BaseballWeightsManager {
    pub weights: Arc<Mutex<Vec<Tensor>>>,
    pub device: Device,
    pub network_map: NetworkMap,
    pub num_conns: Arc<AtomicI32>,
    pub max_conns: i32,
    pub first_moment_map: HashMap<usize, Tensor>,
    pub second_moment_map: HashMap<usize, Tensor>,
    pub learning_rate: f32,
    pub iteration: AtomicI32
}

impl BaseballWeightsManager {
    fn new(
        network_map: NetworkMap,
        dev: Device,
        learning_rate: f32
    ) -> Result<Self> {
        let mut weight: Vec<Tensor> = Vec::new();

        for i in 0..network_map.layers.len(){
            let shape: (usize, usize) = (network_map.layers[i].0 as usize, network_map.layers[i].1 as usize);

            let fan_in: f64 = shape.1 as f64;
            let kaiming_std_dev: f32 = (2.0 / fan_in).sqrt() as f32;
            let layer_init: Tensor = Tensor::randn(
                0.0f32,
                kaiming_std_dev,
                shape,
                &dev
            )?;
            weight.push(layer_init);
        }

        let network_info: &Vec<(i32, i32)> = &network_map.layers;
        let mut first_moment_map: HashMap<usize, Tensor> = HashMap::new();
        let mut second_moment_map: HashMap<usize, Tensor> = HashMap::new();
        for i in 0..network_info.len() {
            let shape: (usize, usize) = (network_info[i].0 as usize, network_info[i].1 as usize);
            let blank_moment: Tensor = Tensor::zeros(shape, candle_core::DType::F32, &dev).expect("Failed to initialize blank moment");
            first_moment_map.insert(i, blank_moment.clone());
            second_moment_map.insert(i, blank_moment.clone());
        }

        Ok(
            Self {
                weights: Arc::new(Mutex::new(weight)),
                device: dev,
                network_map,
                num_conns: Arc::new(AtomicI32::new(0)),
                max_conns: 5,
                first_moment_map,
                second_moment_map,
                learning_rate,
                iteration: AtomicI32::new(1),
            }
        )
    }
}

#[tonic::async_trait]
impl WeightsManager for BaseballWeightsManager {
    async fn update_weights(
        &self,
        request: Request<ModelUpdate>
    ) -> std::result::Result<Response<WeightResponse>, Status> {
        let req_data: ModelUpdate = request.into_inner();
        let network_info: &Vec<(i32, i32)> = &self.network_map.layers;
        println!("Pre layer loop");
        for layer in req_data.layers {
            let mut i: usize = layer.id as usize;
            if i % 2 == 1 {
                continue;
            } else {
                i /= 2;
            }
            let shape: (usize, usize) = (network_info[i].0 as usize, network_info[i].1 as usize);
            println!("Layer id: {}", layer.id);
            println!("Receiced {} gradients.", layer.weights.len());
            println!("Shape dim 1: {}, Shape dim 2: {}", network_info[i].0, network_info[i].1);

            let flat_tensor: Tensor = Tensor::from_vec(layer.weights, shape.0 * shape.1, &self.device).map_err(|e| Status::internal(e.to_string()))?;
            let gradients: Tensor = flat_tensor.reshape(shape).map_err(|e| Status::internal(e.to_string()))?;

            let first_decay_rate: f32 = 0.9;
            let second_decay_rate: f32 = 0.999;

            let current_iter: i32 = self.iteration.load(Ordering::SeqCst);
            let (temp_fm, temp_sm): (Tensor, Tensor) = calculate_moments(
                self.first_moment_map.get(&i).unwrap(),
                self.second_moment_map.get(&i).unwrap(),
                first_decay_rate,
                second_decay_rate,
                &gradients,
                current_iter
            ).map_err(|e| Status::internal(e.to_string()))?;
            let sqrt_sm: Tensor = temp_sm.sqrt().map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            let epsilon: f32 = 1e-8;
            let denom: Tensor = sqrt_sm.affine(1.0, epsilon as f64).map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            let frac: Tensor = (temp_fm / &denom).map_err(|e| Status::internal(e.to_string()))?;
            let adjustment: Tensor = frac.affine(self.learning_rate as f64, 0.0).map_err(|e| Status::internal(e.to_string()))?; 
            let mut state: MutexGuard<Vec<Tensor>> = self.weights.lock().await;
            let new_weights: Tensor = (&state[i] - &adjustment).map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            state[i] = new_weights;  
        }
        println!("Post layer loop");
        self.iteration.fetch_add(1, Ordering::SeqCst);
        //convert layers to Tensor and pass to calculate moments. Also track moments

        let reply: WeightResponse = WeightResponse { success: true };
        Ok(Response::new(reply))
    }
    async fn request_weights(
        &self,
        request: Request<WeightsRequest>
    ) -> std::result::Result<Response<ModelUpdate>, Status> {
        let req_data: WeightsRequest = request.into_inner();
        println!("Worker id: {}", req_data.worker_id);
        
        let state: MutexGuard<Vec<Tensor>> = self.weights.lock().await;
        let mut response_layers: Vec<LayerGradient> = Vec::with_capacity(state.len());

        for (i, layer) in state.iter().enumerate() {
            let layer: &Tensor = layer;
            let flat_tensor: Tensor = layer.flatten_all().map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            let contiguous_tensor: Tensor = flat_tensor.contiguous().map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            let weight_vec: Vec<f32> = contiguous_tensor.to_vec1::<f32>().map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            response_layers.push(LayerGradient {
                id: i as i32,
                weights: weight_vec.clone()
            })
        }

        let reply: ModelUpdate = ModelUpdate { layers: response_layers };
        Ok(Response::new(reply))
    }
    async fn wake_worker(
        &self,
        _request: Request<GoodMorning>
    ) -> std::result::Result<Response<WorkerRegistration>, Status> {
        println!("Num_conns: {:?}, max_conns: {:?}", self.num_conns, self.max_conns);
        let reply: WorkerRegistration = if self.num_conns.load(Ordering::SeqCst) == self.max_conns {
            WorkerRegistration {
                success: false,
                worker_id: -1
            }
        } else {
            let prev_num_conn = self.num_conns.fetch_add(1, Ordering::SeqCst);

            let worker_id = prev_num_conn; //may want to make this pnc + 1
            WorkerRegistration {
                success: true,
                worker_id: worker_id
            }
        };
        Ok(Response::new(reply))
        //TODO: will also need a worker deregistration sort of thing once a work is done w all training
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let network_map: NetworkMap = read_config_from_file("network_map.json")?;
    let dev: Device = Device::new_metal(0).unwrap_or(Device::Cpu);

    let addr: std::net::SocketAddr = "0.0.0.0:50051".parse()?;
    let manager: BaseballWeightsManager = BaseballWeightsManager::new(
        network_map,
        dev,
        0.001
    )?;

    println!("gRPC Server listening on {}", addr);

    Server::builder()
        .add_service(WeightsManagerServer::new(manager))
        .serve(addr)
        .await?;

    Ok(())
}