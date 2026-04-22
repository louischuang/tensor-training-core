# API Authentication And Access-Control Plan

## Goal

Define a practical security plan for exposing Tensor Training Core APIs beyond trusted local development.

This document is intentionally a design guide, not an implemented feature flag. The current repository API is still unauthenticated by default.

## Current State

- FastAPI endpoints are available without built-in authentication.
- The API is suitable for local development, CI smoke checks, and trusted internal lab environments.
- Long-running operations such as training and export can create or expose artifact paths, logs, and model outputs.

## Risks To Address

- Untrusted users triggering expensive training or export jobs
- Unauthorized reads of artifact metadata and training logs
- Uncontrolled access to raw dataset locations or generated model artifacts
- Replay or abuse of async job endpoints
- Accidental public exposure of a development instance

## Recommended Security Model

### Phase A. Basic Gateway Protection

Use one of these deployment patterns before exposing the API beyond localhost:

- private network only
- VPN-only access
- reverse proxy with IP allowlist
- reverse proxy with shared token enforcement

This is the minimum recommended barrier before enabling external users.

### Phase B. Application-Level Token Authentication

Add API token authentication at the FastAPI layer with:

- `Authorization: Bearer <token>`
- token identity mapped to a human owner or service account
- token metadata stored outside the repository source tree
- token rotation support
- token revocation support

Recommended environment variables:

- `TENSOR_TRAINING_CORE_AUTH_MODE=disabled|static_token|proxy_header`
- `TENSOR_TRAINING_CORE_API_TOKENS_FILE=/secure/path/api_tokens.json`
- `TENSOR_TRAINING_CORE_TRUST_PROXY_AUTH=false`

### Phase C. Role-Based Authorization

Define coarse roles first:

- `viewer`
  - can read health, job status, logs, and artifact metadata
- `operator`
  - can run dataset prepare, training, evaluation, and export
- `admin`
  - can retry jobs, manage tokens, and inspect all artifacts

Recommended default endpoint policy:

- `GET /health`: public or viewer
- `GET /training/jobs/{job_id}`: viewer
- `GET /training/jobs/{job_id}/logs`: viewer
- `GET /training/jobs/{job_id}/logs/stream`: viewer
- `GET /artifacts/{job_id}`: viewer
- `POST /datasets/import/coco`: operator
- `POST /datasets/prepare`: operator
- `POST /training/jobs`: operator
- `POST /training/jobs/async`: operator
- `POST /training/jobs/{job_id}/retry`: operator or admin
- `POST /exports/tflite`: operator
- `POST /exports/mobile-bundle`: operator

## Identity And Audit Fields

When authentication is added, extend existing structured logs with:

- `auth_subject`
- `auth_role`
- `auth_mode`
- `request_id`
- `remote_addr`
- `forwarded_for`

Audit requirements:

- persist token subject on every write operation
- attach requester identity to job records
- attach requester identity to retry lineage
- persist denied-access events to API request logs

## Artifact Access Rules

Recommended policy:

- API responses should return artifact metadata, not arbitrary filesystem browsing
- only expose paths already owned by the resolved job record
- keep raw datasets under `data/raw/` outside public API surface
- never expose token files, local environment config, or unrelated repo files

If a future dashboard is added, it should read from the same job and registry records instead of scanning the full filesystem.

## Async And SSE Considerations

Authentication must apply consistently to:

- async training submission
- status polling
- log snapshots
- SSE log streams

Recommended controls:

- deny duplicate async submissions per authenticated subject when appropriate
- cap concurrent training jobs per token or role
- require the same role level for log streaming as for job status access

## Deployment Recommendations

### Local Development

- keep auth disabled
- bind to `127.0.0.1`

### Shared Internal Server

- put the API behind a reverse proxy
- enable static bearer-token authentication
- restrict write operations to operator tokens

### Internet-Exposed Deployment

- require TLS termination at the proxy
- use short-lived service credentials or delegated identity
- add rate limiting
- add request-size limits
- add explicit artifact download policy
- add background worker isolation before enabling public write access

## Suggested Implementation Order

1. Add auth configuration settings and security dependency injection points
2. Implement static bearer-token validation
3. Add role mapping and endpoint guards
4. Add audit fields to API request logs and job records
5. Add integration tests for `401`, `403`, and authorized request flows
6. Document proxy deployment examples

## Non-Goals For The First Security Pass

- OAuth or OpenID Connect
- multi-tenant resource isolation
- fine-grained per-dataset ACLs
- browser login UI

These can wait until the repository is deployed as a shared platform rather than an internal engineering tool.
