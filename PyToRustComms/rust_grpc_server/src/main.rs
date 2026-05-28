use tonic::{transport::Server, Request, Response, Status};

pub mod parameter_server {
    tonic::include_proto!("parameter_server");
}

use parameter_server::weights_manager_server::{WeightsManager, WeightsManagerServer};
use parameter_server::{WeightRequest, WeightResponse};

#[derive(Debug, Default)]
pub struct BaseballWeightsManager {}

#[tonic::async_trait]
impl WeightsManager for BaseballWeightsManager {
    async fn update_weights(
        &self,
        request: Request<WeightRequest>
    ) -> Result<Response<WeightResponse>, Status> {
        let req_data = request.into_inner();
        let grad_count = req_data.gradients.len();

        println!("Received {} gradients.", grad_count);

        let reply = WeightResponse { success: true };
        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let manager = BaseballWeightsManager::default();

    println!("gRPC Server listening on {}", addr);

    Server::builder()
        .add_service(WeightsManagerServer::new(manager))
        .serve(addr)
        .await?;

    Ok(())
}