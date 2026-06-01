FROM rust:slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config ca-certificates libgcc-s1 && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN cargo build --release 2>/dev/null; true
RUN rm -rf src

COPY src/ src/
RUN cargo build --release --target-dir /tmp/target && \
    cp /tmp/target/release/llm-proxy /usr/local/bin/llm-proxy

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/llm-proxy /usr/local/bin/llm-proxy

EXPOSE 7878
ENTRYPOINT ["llm-proxy"]
