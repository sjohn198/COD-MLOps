FROM rust:1.92 AS builder

WORKDIR /usr/app/src

RUN apt-get update && apt-get install -y protobuf-compiler

COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/app/src/target/release/rust_grpc_server /usr/local/bin/rust_grpc_server

COPY best_baseball_predictor.safetensors /app/best_baseball_predictor.safetensors

EXPOSE 50051

CMD ["rust_grpc_server"]