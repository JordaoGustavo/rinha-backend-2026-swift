FROM swift:6.1 AS build

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY Package.swift Package.resolved ./
COPY Sources/CSimd/ Sources/CSimd/
COPY Sources/FraudDetector/ Sources/FraudDetector/
COPY Sources/Preprocessor/ Sources/Preprocessor/
COPY Sources/RinhaApp/ Sources/RinhaApp/
RUN mkdir -p Tests/FraudDetectorTests

RUN swift build -c release --product RinhaApp \
 && cp "$(swift build -c release --show-bin-path)/RinhaApp" /app-binary

FROM swift:6.1-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app

ARG INDEXES_URL=https://github.com/JordaoGustavo/rinha-backend-2026-swift/releases/download/v0.1.0/indexes.tar.gz
ARG RESOURCES_BASE_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources

RUN mkdir -p /data /resources \
 && curl -fsSL "$INDEXES_URL" | tar xz -C /data \
 && curl -fsSL "$RESOURCES_BASE_URL/mcc_risk.json" -o /resources/mcc_risk.json \
 && curl -fsSL "$RESOURCES_BASE_URL/normalization.json" -o /resources/normalization.json

COPY --from=build /app-binary ./RinhaApp
COPY resources/model.bin /resources/model.bin

ENV INDEX_PATH=/data/ivf.bin
ENV EXACT_PATH=/data/exact.bin
ENV RESOURCES_PATH=/resources
ENV API_PORT=8080
ENTRYPOINT ["./RinhaApp"]
