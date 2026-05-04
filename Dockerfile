# syntax=docker/dockerfile:1
FROM rust:alpine AS builder
WORKDIR /app

RUN apk add --no-cache musl-dev

# Cache dependency compilation layer
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN cargo build --release 2>/dev/null; true
RUN rm -rf src

# Build the actual binary
COPY src/ src/
RUN cargo build --release && \
    cp target/release/llm-proxy /llm-proxy

# ── runtime ──────────────────────────────────────────────────────────────
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /llm-proxy /usr/local/bin/llm-proxy

EXPOSE 7878
ENTRYPOINT ["llm-proxy"]
