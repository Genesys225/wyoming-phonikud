FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY wyoming_phonikud/ ./wyoming_phonikud/

VOLUME ["/data"]
EXPOSE 10201

ENTRYPOINT ["python", "-m", "wyoming_phonikud"]
