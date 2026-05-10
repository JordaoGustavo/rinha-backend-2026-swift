#!/usr/bin/env bash
set -euo pipefail

URL="${API_URL:-http://localhost:8181}/fraud-score"
CONCURRENCY="${CONCURRENCY:-10}"
REQUESTS="${REQUESTS:-1000}"

# Varied payloads
PAYLOADS=(
  '{"transaction":{"amount":150,"installments":3,"requested_at":"2026-04-15T14:30:00Z"},"customer":{"avg_amount":100,"tx_count_24h":5,"known_merchants":["m-001"]},"merchant":{"id":"m-001","mcc":"5812","avg_amount":120},"terminal":{"is_online":true,"card_present":true,"km_from_home":10.5},"last_transaction":{"timestamp":"2026-04-15T12:00:00Z","km_from_current":5.0}}'
  '{"transaction":{"amount":9500,"installments":1,"requested_at":"2026-04-15T03:15:00Z"},"customer":{"avg_amount":200,"tx_count_24h":15,"known_merchants":[]},"merchant":{"id":"m-999","mcc":"7995","avg_amount":5000},"terminal":{"is_online":true,"card_present":false,"km_from_home":500},"last_transaction":{"timestamp":"2026-04-14T22:00:00Z","km_from_current":300}}'
  '{"transaction":{"amount":25,"installments":1,"requested_at":"2026-04-15T10:00:00Z"},"customer":{"avg_amount":30,"tx_count_24h":2,"known_merchants":["m-010","m-011"]},"merchant":{"id":"m-010","mcc":"5411","avg_amount":28},"terminal":{"is_online":false,"card_present":true,"km_from_home":0.5},"last_transaction":null}'
  '{"transaction":{"amount":3000,"installments":12,"requested_at":"2026-04-15T19:45:00Z"},"customer":{"avg_amount":500,"tx_count_24h":1,"known_merchants":["m-020"]},"merchant":{"id":"m-021","mcc":"6011","avg_amount":2800},"terminal":{"is_online":true,"card_present":true,"km_from_home":25},"last_transaction":{"timestamp":"2026-04-15T08:00:00Z","km_from_current":15}}'
)

echo "=== Rinha Backend 2026 (Swift) Benchmark ==="
echo "URL: $URL"
echo "Concurrency: $CONCURRENCY"
echo "Total requests: $REQUESTS"
echo ""

# Check if wrk or ab is available
if command -v wrk &>/dev/null; then
    echo "--- Using wrk ---"
    # Write lua script for varied payloads
    cat > /tmp/rinha_bench.lua << 'LUA'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

payloads = {
  '{"transaction":{"amount":150,"installments":3,"requested_at":"2026-04-15T14:30:00Z"},"customer":{"avg_amount":100,"tx_count_24h":5,"known_merchants":["m-001"]},"merchant":{"id":"m-001","mcc":"5812","avg_amount":120},"terminal":{"is_online":true,"card_present":true,"km_from_home":10.5},"last_transaction":{"timestamp":"2026-04-15T12:00:00Z","km_from_current":5.0}}',
  '{"transaction":{"amount":9500,"installments":1,"requested_at":"2026-04-15T03:15:00Z"},"customer":{"avg_amount":200,"tx_count_24h":15,"known_merchants":[]},"merchant":{"id":"m-999","mcc":"7995","avg_amount":5000},"terminal":{"is_online":true,"card_present":false,"km_from_home":500},"last_transaction":{"timestamp":"2026-04-14T22:00:00Z","km_from_current":300}}',
  '{"transaction":{"amount":25,"installments":1,"requested_at":"2026-04-15T10:00:00Z"},"customer":{"avg_amount":30,"tx_count_24h":2,"known_merchants":["m-010","m-011"]},"merchant":{"id":"m-010","mcc":"5411","avg_amount":28},"terminal":{"is_online":false,"card_present":true,"km_from_home":0.5},"last_transaction":null}',
  '{"transaction":{"amount":3000,"installments":12,"requested_at":"2026-04-15T19:45:00Z"},"customer":{"avg_amount":500,"tx_count_24h":1,"known_merchants":["m-020"]},"merchant":{"id":"m-021","mcc":"6011","avg_amount":2800},"terminal":{"is_online":true,"card_present":true,"km_from_home":25},"last_transaction":{"timestamp":"2026-04-15T08:00:00Z","km_from_current":15}}'
}

counter = 0
function request()
    counter = counter + 1
    wrk.body = payloads[(counter % #payloads) + 1]
    return wrk.format(nil, nil, nil, wrk.body)
end
LUA

    wrk -t2 -c"$CONCURRENCY" -d10s -s /tmp/rinha_bench.lua "$URL"

elif command -v ab &>/dev/null; then
    echo "--- Using Apache Bench (ab) ---"
    echo "${PAYLOADS[0]}" > /tmp/rinha_payload.json
    ab -n "$REQUESTS" -c "$CONCURRENCY" -T 'application/json' -p /tmp/rinha_payload.json "$URL"

else
    echo "--- Using curl (sequential, no concurrency tool found) ---"
    echo "Install 'wrk' for proper benchmarking: brew install wrk"
    echo ""

    # Sequential latency measurement
    declare -a LATENCIES
    SAMPLE=100

    echo "Running $SAMPLE sequential requests..."
    for i in $(seq 1 $SAMPLE); do
        IDX=$(( (i - 1) % ${#PAYLOADS[@]} ))
        START=$(date +%s%N 2>/dev/null || perl -MTime::HiRes=time -e 'printf "%.0f\n", time*1e9')
        RESP=$(curl -s -w "\n%{time_total}" -X POST "$URL" \
            -H 'Content-Type: application/json' \
            -d "${PAYLOADS[$IDX]}")
        TIME_TOTAL=$(echo "$RESP" | tail -1)
        LATENCIES+=("$TIME_TOTAL")
    done

    echo ""
    echo "=== Results (sequential, $SAMPLE requests) ==="
    # Sort and compute percentiles
    printf '%s\n' "${LATENCIES[@]}" | sort -n | awk '
    BEGIN { count = 0 }
    {
        vals[count++] = $1 * 1000  # convert to ms
        sum += $1 * 1000
    }
    END {
        printf "  Avg:  %.2f ms\n", sum / count
        printf "  Min:  %.2f ms\n", vals[0]
        printf "  p50:  %.2f ms\n", vals[int(count * 0.50)]
        printf "  p90:  %.2f ms\n", vals[int(count * 0.90)]
        printf "  p99:  %.2f ms\n", vals[int(count * 0.99)]
        printf "  Max:  %.2f ms\n", vals[count - 1]
        printf "  RPS:  %.0f (estimated sequential)\n", 1000 / (sum / count)
    }'
fi
