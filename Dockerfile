# syntax=docker/dockerfile:1
FROM rust:alpine AS builder
WORKDIR /app

RUN apk add --no-cache musl-dev

# Copy everything and build
COPY . .
RUN cargo build --release && \
    cp target/release/llm-proxy /llm-proxy

# ── runtime ──────────────────────────────────────────────────────────────
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /llm-proxy /usr/local/bin/llm-proxy

EXPOSE 7878
ENTRYPOINT ["llm-proxy"]
