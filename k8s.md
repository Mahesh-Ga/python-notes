# Detailed Notes: Kubernetes & Orchestration

---

## 25. Kubernetes & Orchestration

### 25.1 Why orchestration exists
Docker (§16) gives you a single container. Production needs many containers, across many machines, that:
- restart themselves when they crash
- get replaced when the underlying node dies
- scale up/down with load
- get updated without downtime
- discover each other without hardcoded IPs

Kubernetes (k8s): automates the deployment, scaling, and management of containerized applications
- Portable, extensible, open-source platform for managing containerized apps
- Facilitates both declarative configuration and automation
- desired state - I want 3 replicas of image running with this config  
- **controller** - Kubernetes has small loops running inside called controllers to check is desired state same as actual
you need 3 and only 2 are running, it makes a 3rd.
- health checks

What Kubernetes provide?
- Service discovery and load balancing
-  Storage orchestration - k8s allows you to automatically mount a storage system of your choice, such as local storages, public cloud providers, and more
-  Automated rollouts and rollback
- Automatic bin packing
  - You provide Kubernetes with a cluster of nodes that it can use to run containerized tasks
  - You tell Kubernetes how much CPU and memory (RAM) each container needs
  - Kubernetes can fit containers onto your nodes to make the best use of your resources
- Self-healing = Kubernetes restarts containers that fail, replaces containers, kills containers that don’t respond to your user-defined health check
- Secret and configuration management
  - Kubernetes lets you store and manage sensitive information, such as passwords, OAuth tokens, and ssh keys
  - You can deploy and update secrets and application configuration without rebuilding your container images, and without exposing secrets in your stack configuration.

### Kubernetes Cluster

 When you deploy Kubernetes, you get a cluster.
§ A cluster is a set of machines (nodes), that run 
containerized applications managed by Kubernetes
§ A cluster has at least one worker node and at least 
one master node
§ The worker node(s) host the pods that are the 
components of the application
§ The master node(s) manages the worker nodes and 
the pods in the cluster
§ Multiple master nodes are used to provide a cluster 
with failover and high availability--
e--->
pro

### 25.2 Pods — smallest, most basic deployable unit of computing
 Instead of running containers directly, Kubernetes wraps one or more containers into a single logical structure called a Pod. 
 - All containers inside a single Pod share the exact same network space, IP address, port range, and storage volumes
 - Pods are **ephemeral(temporarily) and disposable** — " means they are expected to crash, be terminated, and get replaced constantly without breaking a sweat.

### 25.3 Deployments — desired state for stateless apps
A **Deployment** manages a set of identical Pod replicas via a **ReplicaSet**.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-api
spec:
  replicas: 3
  selector:
    matchLabels: { app: order-api }
  template:
    metadata:
      labels: { app: order-api }
    spec:
      containers:
        - name: order-api
          image: myregistry/order-api:1.4.2
          ports: [{ containerPort: 8000 }]
```

- You edit the Deployment (e.g. bump `image` tag or `replicas`) → Deployment creates a **new ReplicaSet** and manages the rollout between old and new ReplicaSets (see §25.7).
- If a Pod dies (crash, node failure), the ReplicaSet controller notices `actual (2) != desired (3)` and schedules a replacement immediately — this is the self-healing behavior.
- **StatefulSet** is the sibling controller for stateful apps (databases, queues) — gives Pods stable identity/hostname and stable storage across restarts, and manages ordered start/stop. **DaemonSet** runs exactly one Pod per node (log agents, node exporters).

### 25.4 Services — stable networking over unstable Pods
Pods get a new IP every time they're recreated, so nothing should ever talk to a Pod IP directly. A **Service** is a stable virtual IP + DNS name that load-balances across all Pods matching a label selector — it's the piece of glue that makes Deployments (unstable Pod IPs) usable.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: order-api
spec:
  selector: { app: order-api }     # routes to any Pod with this label
  ports: [{ port: 80, targetPort: 8000 }]
  type: ClusterIP                   # default
```

Service types:
- **ClusterIP** (default) — internal-only virtual IP, reachable from inside the cluster. Used for service-to-service traffic.
- **NodePort** — exposes the Service on a static port on every node's IP. Mostly a building block, rarely used directly in prod.
- **LoadBalancer** — provisions an external cloud load balancer (AWS ELB/GCP LB) pointing at the Service. Standard way to expose something to the internet.
- **Ingress** (technically a separate resource, not a Service type) — an HTTP(S) reverse proxy/router (nginx-ingress, ALB controller) that gives you host/path-based routing, TLS termination, and lets many Services share one external load balancer — cheaper and more flexible than one LoadBalancer per Service.

```
Internet -> Ingress -> Service(ClusterIP) -> [Pod, Pod, Pod]  (load-balanced by label selector)
```

### 25.5 ConfigMaps & Secrets — separating config from image
Never bake environment-specific config into the image (§2/§4 config management principle, same idea at cluster scale).
- **ConfigMap**: non-sensitive key/value config (feature flags, log level, external URLs), injected as env vars or mounted as files.
- **Secret**: same idea for sensitive data (API keys, DB passwords, TLS certs). Base64-encoded, not encrypted, by default — **base64 is encoding, not security**. Real protection requires enabling encryption-at-rest for etcd and/or an external secrets manager (Vault, AWS Secrets Manager, Sealed Secrets/External Secrets Operator) so plaintext secrets never live in raw YAML/git.

```yaml
envFrom:
  - configMapRef: { name: order-api-config }
  - secretRef: { name: order-api-secrets }
```

Changing a ConfigMap/Secret does **not** automatically restart Pods to pick up new values (unless mounted as a volume and your app watches for file changes) — a common gotcha. Standard workaround: include a checksum of the config in a Pod template annotation so a config change forces a new rollout.

### 25.6 Liveness vs Readiness probes — and why confusing them causes cascading outages
This is the highest-signal interview topic in this section. Two probes that sound similar but trigger opposite corrective actions:

| | Liveness probe | Readiness probe |
|---|---|---|
| Question it answers | "Is this container stuck/deadlocked and should be killed?" | "Is this Pod currently able to serve traffic?" |
| Action on failure | **kubelet kills and restarts the container** | **Service removes the Pod from its load-balancing endpoints** (Pod keeps running, just stops receiving traffic) |
| Wrong config's failure mode | Restart loop | Silent traffic blackhole (until it recovers) |

```yaml
livenessProbe:
  httpGet: { path: /healthz, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet: { path: /ready, port: 8000 }
  periodSeconds: 5
  failureThreshold: 3
```

**Why confusing them causes cascading outages — the classic failure mode:**
A Pod is temporarily overwhelmed (e.g. DB connection pool exhausted, or it's warming a cache at startup) and its `/health` endpoint starts returning errors because *some dependency* is slow — not because the process itself is deadlocked.

- If that endpoint is wired to **liveness**: kubelet decides the container is dead and **kills it**. But the process wasn't actually broken — it was just waiting on an overloaded downstream. Killing it drops in-flight requests, and the replacement Pod comes up cold (empty cache, cold connection pool) into the *same* overloaded downstream, fails its liveness probe again, gets killed again → **crash-loop**. Meanwhile the remaining Pods now absorb the killed Pod's traffic share, pushing them toward the same overload → more restarts → the whole Deployment can crash-loop in cascade, which is strictly worse than the original slowness.
- The correct wiring: that same "am I healthy" check should be **readiness**, not liveness. Readiness failing just pulls the Pod out of the Service's endpoint list — no traffic, no kill, and it quietly rejoins once the dependency recovers. **Liveness should only check "is my own process alive/not deadlocked"** (e.g. a trivial in-process check, not a check that reaches out to the DB/downstream services) — because liveness failure = destructive action (kill), so it must never depend on something outside the container's own control.
- Rule of thumb: **liveness = "restart me," readiness = "don't send me traffic."** If a check can fail because of something the container *cannot fix by restarting* (a slow DB, a saturated downstream API), it must never be a liveness probe.
- A related probe, **startupProbe**, exists so slow-starting apps (JVM warmup, large migrations) get a longer grace period before liveness starts counting failures — without it you either set liveness's `initialDelaySeconds` too generously (masks real hangs later) or too tight (kills the app mid-startup).

### 25.7 Resource requests & limits — why both matter, differently
```yaml
resources:
  requests: { cpu: "250m", memory: "256Mi" }   # scheduling guarantee
  limits:   { cpu: "500m", memory: "512Mi" }   # hard ceiling
```

- **Requests** are what the scheduler uses to *place* the Pod — it only schedules a Pod onto a node that has that much CPU/memory unreserved. Under-requesting → node gets overcommitted → Pods compete/starve under load. Over-requesting → wastes cluster capacity (Pods reserve resources they never use, other Pods can't be scheduled).
- **Limits** are the hard ceiling the kubelet/kernel enforces at runtime:
  - **CPU limit** exceeded → the container is **throttled** (CFS bandwidth throttling — it doesn't get killed, it just gets paused/slowed within each scheduling period). Silent latency killer if set too tight; easy to misdiagnose as "the app is slow" when it's actually being throttled.
  - **Memory limit** exceeded → the container is **OOMKilled** immediately (`OOMKilled` in `kubectl describe pod`) — no graceful degradation, memory can't be throttled the way CPU can.
- **QoS classes** derive from requests/limits and decide eviction order under node pressure:
  - `Guaranteed` (requests == limits for all containers) — evicted last.
  - `Burstable` (requests < limits) — evicted before Guaranteed.
  - `BestEffort` (no requests/limits set at all) — evicted first.
- Practical guidance: always set both. No limits at all lets one runaway Pod starve/OOM its whole node (noisy neighbor). Setting `requests == limits` (Guaranteed QoS) is the standard choice for latency-sensitive production services precisely to avoid CPU throttling surprises and get eviction priority.

### 25.8 Horizontal Pod Autoscaling (HPA)
The HPA controller adjusts `replicas` on a Deployment automatically based on observed metrics against a target.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: order-api }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: order-api }
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
```

- Default metric is CPU utilization (needs `metrics-server` installed), but custom/external metrics are common in real systems — e.g. **queue depth** or **Kafka consumer lag** (ties directly back to §23.6 backpressure) via the Prometheus Adapter or KEDA, so worker pools scale on "how much work is waiting" rather than CPU, which is often a lagging/misleading signal for I/O-bound workers.
- HPA changes `replicas`; it does **not** change per-Pod resource requests/limits — that's **Vertical Pod Autoscaler (VPA)**, a separate mechanism (and the two don't compose cleanly together on CPU/memory without care — usually pick one axis per Deployment).
- Scale-up reacts fast by default; scale-down has a stabilization window (default 5 min) to avoid flapping (rapid scale up/down thrashing) when load is bursty near the threshold.
- HPA is only as good as the underlying probes/requests — if readiness probes are wrong, new Pods can be counted as "capacity" before they can actually serve traffic, undermining the whole point of scaling up.

### 25.9 Rolling update vs blue-green vs canary
All three answer "how do I ship a new version without downtime" — the difference is blast radius and rollback speed vs. infrastructure cost.

**Rolling update** (Deployment's default strategy):
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1   # how many old Pods can be down at once
    maxSurge: 1          # how many extra new Pods can exist above `replicas` during rollout
```
Gradually replaces old Pods with new ones, a few at a time, keeping the Service serving traffic throughout (mix of old+new versions live simultaneously during the rollout). Cheap (no extra infra), zero downtime if readiness probes are correct (a new Pod isn't added to the Service's endpoints until it passes readiness — this is *why* §25.6 matters so much here specifically). Downside: **both versions run at once**, so it requires the new version to be backward compatible with anything shared (DB schema, message formats) during the overlap window; rollback means rolling the same process in reverse, which isn't instant.

**Blue-green**: run two complete, independent environments ("blue" = current prod, "green" = new version), fully deployed and warmed up, then flip the Service/router to send 100% of traffic to green in one atomic switch. Rollback = flip back to blue instantly. Simplest mental model, fastest rollback, no mixed-version traffic — but 2x infrastructure cost while both stacks are up, and it's all-or-nothing (a bug only visible under production load hits 100% of users at once, just very briefly since rollback is instant).

**Canary**: route a small percentage of real traffic (e.g. 5%) to the new version alongside the old, watch error rates/latency, then progressively increase the percentage (5% → 25% → 100%) if metrics stay healthy — or abort and route back to 0% if they don't. Needs a router capable of weighted traffic splitting (Ingress controller, service mesh like Istio/Linkerd, or a cloud LB) — plain Kubernetes Deployments don't natively do percentage-based splitting themselves. Best blast-radius control (a bad release only hurts a small slice of users, and you find out *before* full exposure) at the cost of the most operational complexity (traffic-shaping infra, careful metric-based gating, often automated via a progressive-delivery tool like Argo Rollouts/Flagger).

```
Rolling:    [old][old][old] -> [old][old][new] -> [old][new][new] -> [new][new][new]
                                   (mixed traffic throughout — needs backward-compatible versions)

Blue-Green: [blue: 100% traffic]  [green: warmed, 0% traffic]
                    |  instant switch  |
            [blue: 0%]            [green: 100% traffic]   (rollback = flip back, instant)

Canary:     [old: 95%][new: 5%] -> [old: 75%][new: 25%] -> [old: 0%][new: 100%]
                                    (abort at any stage if error rate/latency degrades)
```

| | Infra cost | Rollback speed | Blast radius on bad release | Complexity |
|---|---|---|---|---|
| Rolling update | Low (no duplicate stack) | Slow-ish (re-rollout) | Up to 100%, ramps in | Low — built into Deployment |
| Blue-green | High (2x stack while both live) | Instant (flip router) | 100%, but very briefly | Medium |
| Canary | Medium (small extra capacity) | Fast (route back to 0%) | Small % contained | High (needs traffic splitting + metrics gating) |

### 25.10 Quick interview-answer shape
- "Liveness answers 'should this container be killed and restarted,' readiness answers 'should this Pod receive traffic right now' — wiring a downstream-dependency check to liveness instead of readiness turns a transient slowdown into a kill-and-restart-loop, which sheds capacity from the remaining Pods and can cascade into a full outage. Liveness should only ever check the process's own internal health."
- "Requests are what the scheduler uses to place Pods and guarantee resources; limits are the runtime hard ceiling — CPU limits throttle (slow, no crash), memory limits OOMKill (hard crash). Setting requests==limits gives Guaranteed QoS, which avoids throttling surprises and is evicted last under node pressure."
- "HPA scales replica count off a metric against a target — CPU by default, but queue depth or consumer lag for worker pools is usually the more honest signal, same idea as backpressure-driven autoscaling for Celery/Kafka workers."
- "Rolling update is the cheap default and is safe as long as the new version is backward-compatible during the mixed-version window; blue-green trades 2x infra cost for an instant, all-or-nothing rollback; canary trades operational complexity for the smallest blast radius by proving the new version on a slice of real traffic before going to 100%."
