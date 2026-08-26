# Production Engineering with Python — Full Notes (Basics → Advanced)
###  Interview Questions based on resume.tex and job_description.txt

---

# PART 1: Topics — Basics to Advanced

## BASICS

### 1. Core Python for Production Code
Writing production Python is different from writing scripts: you need clean error handling (specific exceptions, not bare `except:`), context managers (`with` blocks) for resource cleanup (files, DB connections, locks), decorators for cross-cutting concerns (timing, retries, auth checks), and type hints (`typing`, `Pydantic`) so bugs surface before runtime. Know the difference between mutable/immutable defaults (the classic mutable-default-argument bug), generators for memory-efficient iteration, and `__slots__`/dataclasses for lean objects.

### 2. Virtual Environments & Dependency Management
Every production service needs isolated, reproducible dependencies. Know `venv`/`virtualenv` basics, and a lockfile-based tool (Poetry, pip-tools, or `requirements.txt` with pinned versions + hashes). Understand why unpinned dependencies cause "works on my machine" and supply-chain risk, and how lockfiles give reproducible builds across dev/CI/prod.

### 3. Logging Fundamentals
Use Python's `logging` module, not `print()`. Understand log levels (DEBUG/INFO/WARNING/ERROR/CRITICAL), why you set different levels per environment, and log formatting (timestamps, module name, level). This is the foundation for everything in "Observability" below.

### 4. Configuration Management
Separate config from code (12-factor app principle): environment variables, `.env` files for local dev, and a secrets manager (Azure Key Vault, AWS Secrets Manager) for production credentials. Never hardcode secrets or commit them to git. Use Pydantic `BaseSettings` (or similar) to validate config at startup and fail fast on missing/invalid values.

### 5. Error Handling & Exceptions
Design a clear exception hierarchy for your domain (e.g., `ValidationError`, `NotFoundError`, `UpstreamServiceError`). Decide what should crash the process vs. be caught and logged vs. be retried. Understand how unhandled exceptions propagate in async code differently than sync code.

### 6. Testing Basics
`pytest` fundamentals: fixtures, parametrization, mocking (`unittest.mock`) for external calls (DB, HTTP, LLM APIs). Understand the test pyramid — many fast unit tests, fewer integration tests, minimal end-to-end tests — and why untested code is a production liability, not a style preference.

### 7. Running a Web Service (WSGI/ASGI)
Know the difference between WSGI (Flask, sync) and ASGI (FastAPI, async) and why ASGI enables handling many concurrent I/O-bound requests on one process. Know how a production app is actually run: `uvicorn`/`gunicorn` with worker processes, not `python app.py`.

---

## INTERMEDIATE

### 8. REST API Design in Practice
Resource-oriented URLs, correct HTTP verbs/status codes, request/response validation via Pydantic models, pagination for large collections, versioning strategy (URL or header based), and idempotency keys for unsafe operations (e.g., "create this order" should be safely retryable). Authentication (JWT, API keys, OAuth2) and authorization (role/permission checks) belong here too — you implemented JWT auth in your Online Food Ordering project.

### 9. Databases & ORMs in Production
SQLAlchemy/psycopg2 usage patterns, connection pooling (why you never open a raw connection per request), transactions and isolation levels, and schema migrations (Alembic) that are safe to run against a live database (additive changes first, backfill, then remove old columns — avoid locking large tables).

### 10. Caching with Redis
Cache-aside vs write-through patterns, TTL selection, cache invalidation (the classic "two hard problems in CS"), using Redis for session storage (as you did), rate limiting (token bucket via Redis), and pub/sub for lightweight messaging. Know Redis data structures (strings, hashes, sorted sets, lists) and when each fits.

### 11. Background Jobs & Task Queues (Celery)
Producer/broker/worker/result-backend architecture. Retry policies with exponential backoff, task idempotency (critical — a task may run more than once), dead-letter handling for tasks that repeatedly fail, and avoiding long-running tasks blocking a worker pool. You used this directly in your PO→SO and Document Classification services — know the failure modes you handled.

### 12. Concurrency Models & the GIL
Understand the GIL's effect: threads help for I/O-bound work but not CPU-bound work; `multiprocessing` sidesteps the GIL for CPU-bound work by using separate processes; `asyncio` gives high-concurrency I/O without threads at all, but a single blocking call stalls the whole event loop. Knowing when to reach for which is a common interview probe.

### 13. Structured Logging & Correlation IDs
JSON-formatted logs with a `request_id`/`trace_id` that flows through every service and every async task, so a single user request (or agent run) can be reconstructed across logs after the fact. You built this for multi-agent pipelines — be ready to explain how a trace ID moved between agent steps.

### 14. Testing Strategy at Service Level
Integration tests against a real (or containerized/test) database, contract tests for API consumers, mocking third-party APIs and LLM calls deterministically, and — specific to your experience — evaluation pipelines for non-deterministic outputs (LLM/agent responses), where "correctness" is graded rather than asserted exactly.

### 15. Packaging & Reproducible Builds
Pinning exact versions in CI, building once and promoting the same artifact through environments (build once, deploy everywhere) rather than rebuilding per environment, which avoids "it built differently in prod" bugs.

### 16. Containerization (Docker)
Multi-stage builds to keep images small, minimizing layers, non-root users in containers, `.dockerignore`, health checks (`HEALTHCHECK`), and separating build-time vs run-time dependencies. Know how your FastAPI/Celery services were actually packaged and deployed.

### 17. CI/CD Basics
Pipeline stages (lint → test → build → containerize → push → deploy), why fast feedback loops matter, and basic deployment strategies (rolling deploy) before diving into advanced ones below.

---

## ADVANCED

### 18. Observability: Metrics, Logs, Traces
The three pillars: metrics (aggregatable numbers — request rate, error rate, latency percentiles), logs (discrete events), and distributed traces (a request's full path across services/agents, e.g. via OpenTelemetry). You have hands-on experience instrumenting multi-agent pipelines this way — be ready to describe what you measured and how you used it to catch regressions.

### 19. Performance Profiling & Optimization
`cProfile`/`py-spy` for CPU profiling, `memory_profiler`/`tracemalloc` for memory, identifying N+1 query problems, and knowing when the fix is code-level (algorithm, query) vs. infrastructure-level (caching, scaling) vs. architecture-level (async, batching).

### 20. Resilience Patterns
Timeouts on every external call (never wait forever), retries with exponential backoff + jitter (to avoid synchronized retry storms), circuit breakers (stop calling a failing dependency for a cooldown period), and bulkheads (isolate failures so one slow dependency doesn't exhaust all workers/threads).

### 21. Scaling Strategies
Horizontal scaling requires statelessness (session state externalized to Redis, as you did), load balancing, autoscaling based on real signals (queue depth, CPU, latency — not just CPU alone), and understanding what actually becomes the bottleneck first (DB connections are usually the real ceiling, not app servers).

### 22. Databases at Scale
Index design and reading `EXPLAIN ANALYZE` output, read replicas for read-heavy workloads, connection pool sizing math (pool size vs. number of app instances vs. DB max connections), and zero-downtime migration patterns for large tables.

### 23. Message Queues & Event-Driven Architecture
Beyond Celery: Kafka/RabbitMQ for decoupling services, at-least-once vs exactly-once delivery semantics (exactly-once is mostly a myth — design consumers to be idempotent instead), dead-letter queues, and backpressure handling when producers outpace consumers.

### 24. Security in Production
Secrets management (never in env files committed to git), dependency vulnerability scanning, input validation at every trust boundary, least-privilege service credentials, and auth token handling (JWT expiry/refresh, avoiding storing sensitive data in tokens).

### 25. Kubernetes & Orchestration
Deployments, Services, ConfigMaps/Secrets, liveness vs readiness probes (and why confusing them causes cascading outages), resource requests/limits, Horizontal Pod Autoscaling, and rolling updates vs blue-green vs canary deployments.

### 26. Incident Response & On-Call Practices
Alerting on symptoms (user-facing errors/latency) not just causes, runbooks for common failures, blameless postmortems, and rollback-first mentality (roll back before you root-cause under pressure).

### 27. Progressive Rollout & Feature Flags
Canary releases, blue-green deployments, and feature flags to decouple deploy from release — deploy dark, enable gradually, and roll back a feature without a redeploy.

### 28. Evaluation Pipelines for Non-Deterministic (AI/Agent) Systems
This is your standout advanced topic. Unlike traditional software where tests assert exact outputs, agent/LLM systems need graded evaluation: golden datasets, regression detection across model/prompt/agent-graph changes, and quality metrics beyond pass/fail. You built this in production — it's rare and directly matches what Potpie's JD calls out under "observability, evaluation pipelines, agent reliability."

### 29. Cost & Capacity Planning
Understanding what drives infrastructure cost (LLM token usage, DB compute, idle containers) and designing with cost awareness — batching, caching LLM calls, right-sizing containers/worker counts instead of over-provisioning by default.

---

# PART 2: Possible Interview Questions (based on resume.tex + job_description.txt)

## A. Production Engineering / Python
1. Walk me through what happens end-to-end when a request hits one of your FastAPI services in production.
2. How do you make a Celery task idempotent? What happens if the same task runs twice?
3. Async vs sync in FastAPI — when do you choose which, and what breaks if you accidentally block the event loop?
4. How do you decide what becomes a log line vs a metric vs a trace span?
5. Describe a production incident you debugged — how did you find the root cause?
6. How do you handle secrets and config differences across dev/staging/prod/customer on-prem environments?
7. How would you add retries to a call to an external API safely (without causing a retry storm)?

## B. Backend Fundamentals / System Design
8. Design the PO→SO matching service from scratch — how do you compute and calibrate a confidence score, and what happens right at the threshold boundary?
9. When would you choose Neo4j over PostgreSQL for a feature? Walk through the data layer of your NL2SQL/RAG chatbot.
10. How would you design an evaluation pipeline to catch regressions in a multi-agent system before release? (They'll want your real answer — you built one.)
11. How would you design ingestion of a 10-million-line enterprise monorepo into a knowledge graph? (This is close to Potpie's actual product — expect it.)
12. How would you scale a service that suddenly needs to handle 10x traffic?
13. How do you run a schema migration on a live production database with zero downtime?

## C. Databases & Infra
14. Explain Redis eviction policies and how you picked TTLs for session caching.
15. How do you debug a slow PostgreSQL query in production?
16. What happens if a Celery worker crashes mid-task, and how do you avoid duplicated side effects?
17. What's your approach to connection pool sizing across multiple app instances hitting one database?

## D. Docker / Cloud / Deployment
18. Walk through your Azure deployment pipeline for one of your services.
19. What goes into your Dockerfile for a production Python service, and why?
20. How would you deploy the same service into a customer's self-hosted or air-gapped environment vs. your own cloud?

## E. AI Agents / LLM / RAG (expect a deep dive — this is your differentiator and Potpie's core business)
21. Compare LangGraph and CrewAI — why did you choose one over the other for a given project?
22. How do you detect and reduce hallucination in a RAG chatbot answering from both unstructured docs and structured data?
23. How did you actually measure the "~40% accuracy improvement" in your document classification system?
24. How would you extend your agent-evaluation approach to Potpie's context — agents operating over customer codebases instead of internal documents?
25. How do you prevent an LLM-generated SQL query from doing something destructive before it executes?
26. How do you manage cost and latency across a pipeline that makes many chained LLM calls?

## F. Resume Deep-Dives
27. PO→SO Automation: architecture, confidence scoring, failure modes, what happens on an ambiguous/bad match.
28. SDLC Automation product: what does each agent do, how do they hand off state, and how does your observability/eval setup catch regressions across releases?
29. NL2SQL chatbot: how are generated queries validated before running against the database?
30. HRMS & DevOps agents: how does session state work across containerized services?
31. Online Food Ordering System (personal project): JWT auth flow and Redis-backed session design.

## G. Role Fit / Behavioral (Forward-Deployed Engineer specific)
32. Why forward-deployed engineering instead of staying in a pure product engineering role?
33. Tell me about a time you had to understand a system you didn't build, quickly, under pressure.
34. Describe a time requirements were ambiguous or changed mid-project — what did you do?
35. How comfortable are you being on-site at a customer, debugging live issues in front of them?
36. Tell me about a time you disagreed with a stakeholder's technical request — how did you handle it?
37. Tell me about a time you owned something end-to-end with no existing playbook.
38. How do you explain a technical tradeoff to a non-technical stakeholder?

## H. Questions to Ask Potpie
- What does a typical customer discovery-to-production-rollout cycle look like end to end?
- How does the team currently measure agent reliability/quality across different customer deployments?
- How much of this role is on-site vs. remote, and how is that decided per engagement?
- What does the knowledge-graph construction pipeline look like today, and what's the biggest technical bottleneck?
- How does the team balance building reusable product capability vs. one-off customer customization?
