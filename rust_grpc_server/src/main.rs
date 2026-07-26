use tonic::{transport::Server, Request, Response, Status};
use parameter_server::weights_manager_server::{WeightsManager, WeightsManagerServer};
use parameter_server::{WeightResponse, WeightsRequest, LayerGradient, ModelUpdate, GoodMorning, WorkerRegistration};
use std::fs::File;
use std::collections::{ HashMap };
use std::sync::atomic::Ordering;
use candle_core::{Device, Tensor, Result};
use tokio::sync::MutexGuard;
use std::path::{Path, PathBuf};
use std::io::BufReader;
use chrono::{Local, DateTime};
use std::fs;
use rust_grpc_server::baseball_weights_manager::BaseballWeightsManager;
use rust_grpc_server::network_map::NetworkMap;


pub mod parameter_server {
    tonic::include_proto!("parameter_server");
}

fn read_config_from_file<P: AsRef<Path>>(path: P) -> std::result::Result<NetworkMap, Box<dyn std::error::Error>> {
    let file: File = File::open(path)?;
    let buffer: BufReader<File> = BufReader::new(file);
    let network_map: NetworkMap = serde_json::from_reader(buffer)?;

    Ok(network_map)
}

fn calculate_moments(
    first_moment: &Tensor, 
    second_moment: &Tensor, 
    first_decay_rate: f32,
    second_decay_rate: f32,
    gradients: &Tensor,
    iteration: i32) ->  Result<(Tensor, Tensor, Tensor, Tensor)>{
        let decayed_gradients: Tensor = gradients.affine((1.0 - first_decay_rate) as f64, 0.0)?;
        let decayed_first_moment: Tensor = first_moment.affine(first_decay_rate as f64, 0.0)?;
        let new_first_moment: Tensor = (decayed_first_moment + &decayed_gradients)?;

        let squared_gradients: Tensor = (gradients * gradients)?;
        let decayed_squared_gradients: Tensor = squared_gradients.affine((1.0 - second_decay_rate) as f64, 0.0)?;
        let decayed_second_moment: Tensor = second_moment.affine(second_decay_rate as f64, 0.0)?;
        let new_second_moment: Tensor = (decayed_second_moment + &decayed_squared_gradients)?;

        let bias_corrected_first_moment: Tensor = new_first_moment.affine((1.0 / (1.0 - first_decay_rate.powi(iteration))) as f64, 0.0)?;
        let bias_corrected_second_moment: Tensor = new_second_moment.affine((1.0 / (1.0 - second_decay_rate.powi(iteration))) as f64, 0.0)?;

        Ok((
            new_first_moment,
            new_second_moment,
            bias_corrected_first_moment, 
            bias_corrected_second_moment
        ))
}

#[tonic::async_trait]
impl WeightsManager for BaseballWeightsManager {
    async fn update_weights(
        &self,
        request: Request<ModelUpdate>
    ) -> std::result::Result<Response<WeightResponse>, Status> {
        let req_data: ModelUpdate = request.into_inner();
        let network_info: &Vec<(i32, i32)> = &self.network_map.layers;
        //println!("Pre layer loop");
        let mut state: MutexGuard<Vec<Tensor>> = self.weights.lock().await;
        let mut fm_guard: MutexGuard<HashMap<usize, Tensor>>= self.first_moment_map.lock().await;
        let mut sm_guard: MutexGuard<HashMap<usize, Tensor>> = self.second_moment_map.lock().await;
        for layer in req_data.layers {
            let i: usize = layer.id as usize;
            let network_idx: usize = i / 2;

            let in_dim: usize = network_info[network_idx].0 as usize;
            let out_dim: usize = network_info[network_idx].1 as usize;
            let shape: Vec<usize> = if i % 2 == 1 {
                vec![out_dim]
            } else {
                vec![in_dim, out_dim]
            };
            // println!("Layer id: {}", layer.id);
            // println!("Receiced {} gradients.", layer.weights.len());
            // println!("Shape dim 1: {}, Shape dim 2: {}", network_info[network_idx].0, network_info[network_idx].1);

            let flat_tensor: Tensor = Tensor::from_vec(layer.weights.clone(), layer.weights.len(), &self.device).map_err(|e| Status::internal(e.to_string()))?;
            let gradients: Tensor = flat_tensor.reshape(shape).map_err(|e| Status::internal(e.to_string()))?;

            let first_decay_rate: f32 = 0.9;
            let second_decay_rate: f32 = 0.999;

            let current_iter: i32 = self.iteration.load(Ordering::SeqCst);
            let (raw_fm, raw_sm, corr_fm, corr_sm): (Tensor, Tensor, Tensor, Tensor) = calculate_moments(
                fm_guard.get(&i).unwrap(),
                sm_guard.get(&i).unwrap(),
                first_decay_rate,
                second_decay_rate,
                &gradients,
                current_iter
            ).map_err(|e| Status::internal(e.to_string()))?;
            fm_guard.insert(i, raw_fm);
            sm_guard.insert(i, raw_sm);
            let sqrt_sm: Tensor = corr_sm.sqrt().map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            let epsilon: f32 = 1e-8;
            let denom: Tensor = sqrt_sm.affine(1.0, epsilon as f64).map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            let frac: Tensor = (corr_fm / &denom).map_err(|e| Status::internal(e.to_string()))?;
            let adjustment: Tensor = frac.affine(self.learning_rate as f64, 0.0).map_err(|e| Status::internal(e.to_string()))?; 
            let new_weights: Tensor = (&state[i] - &adjustment).map_err(|e: candle_core::Error| Status::internal(e.to_string()))?;
            state[i] = new_weights;  
        }
        //println!("Post layer loop");
        self.iteration.fetch_add(1, Ordering::SeqCst);
        //convert layers to Tensor and pass to calculate moments. Also track moments

        let reply: WeightResponse = WeightResponse { success: true };
        Ok(Response::new(reply))
    }
    async fn request_weights(
        &self,
        request: Request<WeightsRequest>
    ) -> std::result::Result<Response<ModelUpdate>, Status> {
        let _req_data: WeightsRequest = request.into_inner();
        //println!("Worker id: {}", req_data.worker_id);
        
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
        let path_string: String = match self.path.to_str() {
            Some(s) => s.to_string(),
            None => panic!("Path contains invalid UTF-8 characters")
        };
        let reply: WorkerRegistration = if self.num_conns.load(Ordering::SeqCst) == self.max_conns {
            WorkerRegistration {
                success: false,
                worker_id: -1,
                path: path_string
            }
        } else {
            let prev_num_conn = self.num_conns.fetch_add(1, Ordering::SeqCst);

            let worker_id = prev_num_conn; //may want to make this pnc + 1
            WorkerRegistration {
                success: true,
                worker_id: worker_id,
                path: path_string
            }
        };
        Ok(Response::new(reply))
        //TODO: will also need a worker deregistration sort of thing once a work is done w all training
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let network_map: NetworkMap = read_config_from_file("/app/network_map.json")?;
    //for debugging purposes
    let layers: &Vec<(i32, i32)> = &network_map.layers;
    for layer in layers {
        println!("Current layer: {}, {}", layer.0, layer.1);
    }

    let now: DateTime<Local> = Local::now();
    let timestamp_string: String = now.format("%Y-%m-%d %H:%M:%S").to_string();
    let path: PathBuf = Path::new("/app/results").join(&timestamp_string);
    fs::create_dir_all(&path)?;

    let dev: Device = Device::new_metal(0).unwrap_or(Device::Cpu);

    let addr: std::net::SocketAddr = "0.0.0.0:50051".parse()?;
    let manager: BaseballWeightsManager = BaseballWeightsManager::new(
        network_map,
        dev,
        0.001,
        path
    )?;

    println!("gRPC Server listening on {}", addr);

    Server::builder()
        .add_service(WeightsManagerServer::new(manager))
        .serve(addr)
        .await?;

    Ok(())
}