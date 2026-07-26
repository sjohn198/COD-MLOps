use candle_core::{Device, Tensor, Result};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::sync::atomic::AtomicI32;
use std::collections::HashMap;
use crate::network_map::NetworkMap;
use std::path::PathBuf;

#[derive(Debug)]
pub struct BaseballWeightsManager {
    pub weights: Arc<Mutex<Vec<Tensor>>>,
    pub device: Device,
    pub network_map: NetworkMap,
    pub num_conns: Arc<AtomicI32>,
    pub max_conns: i32,
    pub first_moment_map: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub second_moment_map: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub learning_rate: f32,
    pub iteration: AtomicI32,
    pub path: PathBuf
}

impl BaseballWeightsManager {
    pub fn new(
        network_map: NetworkMap,
        dev: Device,
        learning_rate: f32,
        path: PathBuf
    ) -> Result<Self> {
        let mut weight: Vec<Tensor> = Vec::new();
        let network_info: &Vec<(i32, i32)> = &network_map.layers;
        let mut first_moment_map: HashMap<usize, Tensor> = HashMap::new();
        let mut second_moment_map: HashMap<usize, Tensor> = HashMap::new();
        let mut current_idx = 0;

        for i in 0..network_info.len() {
            let in_dim: usize = network_info[i].0 as usize;
            let out_dim: usize = network_info[i].1 as usize;

            let w_shape = (in_dim, out_dim);
            let w_std_dev: f32 = (2.0 / (out_dim as f64)).sqrt() as f32;
            
            let w_tensor = Tensor::randn(0.0f32, w_std_dev, w_shape, &dev)?;
            weight.push(w_tensor);

            let w_moment = Tensor::zeros(w_shape, candle_core::DType::F32, &dev).unwrap();
            first_moment_map.insert(current_idx, w_moment.clone());
            second_moment_map.insert(current_idx, w_moment.clone());
            
            current_idx += 1;

            let b_shape = out_dim;
            let b_std_dev: f32 = (2.0 / (out_dim as f64)).sqrt() as f32;
            
            let b_tensor = Tensor::randn(0.0f32, b_std_dev, b_shape, &dev)?; 
            weight.push(b_tensor);

            let b_moment = Tensor::zeros(b_shape, candle_core::DType::F32, &dev).unwrap();
            first_moment_map.insert(current_idx, b_moment.clone());
            second_moment_map.insert(current_idx, b_moment.clone());
            
            current_idx += 1;
        }

        Ok(
            Self {
                weights: Arc::new(Mutex::new(weight)),
                device: dev,
                network_map,
                num_conns: Arc::new(AtomicI32::new(0)),
                max_conns: 5,
                first_moment_map: Arc::new(Mutex::new(first_moment_map)),
                second_moment_map: Arc::new(Mutex::new(second_moment_map)),
                learning_rate,
                iteration: AtomicI32::new(1),
                path: path
            }
        )
    }
}