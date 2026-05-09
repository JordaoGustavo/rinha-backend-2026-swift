.PHONY: build docker-build docker-up docker-down clean k6

COMPOSE := docker compose -f docker/docker-compose.yml --project-directory docker
IMAGE   := rinha/api-swift:latest

build:
	swift build -c release

docker-build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-up:
	$(COMPOSE) up -d
	@echo "Waiting for /ready..."
	@for i in $$(seq 1 120); do \
		curl -sf http://localhost:9999/ready > /dev/null 2>&1 && echo "  ready in $${i}s" && break; \
		sleep 1; \
		[ $$i -eq 120 ] && echo "  TIMEOUT" && exit 1 || true; \
	done

docker-down:
	$(COMPOSE) down

K6_VUS      ?= 20
K6_DURATION ?= 60s

k6:
	docker run --rm --network host \
		-v $(CURDIR)/scripts/k6:/scripts:ro \
		-e API_URL="http://localhost:9999" \
		-e VUS="$(K6_VUS)" \
		-e DURATION="$(K6_DURATION)" \
		grafana/k6 run /scripts/bench.js

clean:
	rm -rf .build/ data/
