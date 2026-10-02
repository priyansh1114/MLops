# Verification screenshots

These screenshots were captured from the live Docker API on 2 October 2026:

- `docker-api-docs.png` — Swagger UI at `http://127.0.0.1:8000/docs`.
- `docker-api-health.png` — readiness response from `/health`.
- `docker-api-metrics.png` — Prometheus metrics from `/metrics`.

The container `heart-disease-api-local` was running during capture. Kubernetes evidence was captured after the Docker Desktop Kubernetes rollout:

- `kubernetes-rollout.png` — live `kubectl rollout status`, deployment, ready pods, and LoadBalancer service.
- `kubernetes-api-docs.png` — Swagger UI served through Kubernetes port-forward at `http://127.0.0.1:8080/docs`.
- `kubernetes-api-health.png` — readiness response through the Kubernetes Service.
- `kubernetes-api-metrics.png` — Prometheus metrics through the Kubernetes Service.

`kubernetes-rollout.html` contains the captured `kubectl` command output used to render the rollout screenshot. The public cloud URL and user-recorded pipeline video are not available.
