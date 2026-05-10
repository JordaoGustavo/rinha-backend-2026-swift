FROM swift:6.1 AS build
ARG RESOURCES_BASE_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN mkdir -p /resources \
 && curl -fsSL "$RESOURCES_BASE_URL/references.json.gz" -o /resources/references.json.gz \
 && curl -fsSL "$RESOURCES_BASE_URL/mcc_risk.json"      -o /resources/mcc_risk.json \
 && curl -fsSL "$RESOURCES_BASE_URL/normalization.json"  -o /resources/normalization.json

COPY Package.swift Package.resolved ./
COPY Sources/CSimd/ Sources/CSimd/
COPY Sources/FraudDetector/ Sources/FraudDetector/
COPY Sources/Preprocessor/ Sources/Preprocessor/
COPY Sources/RinhaApp/ Sources/RinhaApp/
RUN mkdir -p Tests/FraudDetectorTests

RUN swift build -c release --target Preprocessor

RUN mkdir -p /data \
 && .build/release/Preprocessor /resources/references.json.gz /data/ivf.bin 4096 20 ivf 8 \
 && .build/release/Preprocessor /resources/references.json.gz /data/exact.bin 0 0 exact 0 \
 && .build/release/Preprocessor gen-warmup /resources/warmup-payloads.ndjson 200

RUN swift build -c release --target RinhaApp

FROM swift:6.1-slim
WORKDIR /app
COPY --from=build /src/.build/release/RinhaApp .
COPY --from=build /resources/mcc_risk.json          /resources/mcc_risk.json
COPY --from=build /resources/normalization.json      /resources/normalization.json
COPY --from=build /resources/warmup-payloads.ndjson  /resources/warmup-payloads.ndjson
COPY --from=build /data/ivf.bin                      /data/ivf.bin
COPY --from=build /data/exact.bin                    /data/exact.bin

ENV INDEX_PATH=/data/ivf.bin
ENV EXACT_PATH=/data/exact.bin
ENV RESOURCES_PATH=/resources
ENV API_PORT=8080
ENTRYPOINT ["./RinhaApp"]
