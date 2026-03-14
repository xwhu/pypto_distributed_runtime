# High-Level Design v1: PyPTO Distributed Runtime (CPU + NPU)

## 1. Context and Goals

PyPTO aims to run heterogeneous AI workloads across distributed infrastructure where both CPU and NPU resources are present. Typical target scenarios include:

- LLM serving
- Reinforcement Learning (RL)
- Continuous learning / online adaptation
- Multimodal pipelines (text + vision + audio)
- Recommendation and ranking systems

### Primary goals

1. Provide a unified runtime for mixed CPU/NPU execution.
2. Optimize latency and throughput under real-world workload variability.
3. Support fault-tolerant distributed operation across nodes.
4. Keep programming and deployment interfaces simple enough for iterative adoption.

### Non-goals (v1)

- Full auto-tuning of every operator placement strategy.
- Deep hardware-specific kernel authoring.
- Full global optimal scheduling across all nodes and all jobs.

---

## 2. Workload Characteristics and Why CPU+NPU Mix Is Needed

## 2.1 LLM Serving

- **NPU-heavy:** attention, MLP, dense matrix compute.
- **CPU-heavy:** tokenization, dynamic batching, request routing, cache management, sampling logic, response post-processing.
- **Design implication:** maximize NPU utilization while preserving low tail-latency through smart CPU orchestration.

## 2.2 Reinforcement Learning

- **NPU-heavy:** policy/value model forward+backward.
- **CPU-heavy:** environment stepping (often branch-heavy), replay buffer operations, trajectory assembly.
- **Design implication:** decouple actor/environment pipelines from learner pipelines with asynchronous queues.

## 2.3 Continuous Learning

- **NPU-heavy:** periodic fine-tuning / adaptation passes.
- **CPU-heavy:** data validation, feature engineering, drift checks, gating logic.
- **Design implication:** runtime must co-host inference and adaptation jobs with policy-driven isolation.

## 2.4 Multimodal Pipelines

- **NPU-heavy:** encoder/decoder networks for modalities.
- **CPU-heavy:** media decoding, synchronization, business logic, protocol integration.
- **Design implication:** graph-level partitioning and efficient transfer pipelines are critical.

## 2.5 Recommendation / Ranking

- **NPU-heavy:** dense towers and embedding interaction blocks.
- **CPU-heavy:** candidate generation, joins, filtering, rule systems.
- **Design implication:** combine feature/dataflow stages (CPU) with dense scoring stages (NPU), minimizing cross-device copy overhead.

---

## 3. Architectural Principles

1. **Heterogeneous by default**  
   Runtime assumes every workload may span CPU and NPU.

2. **Asynchronous everywhere**  
   Scheduling, execution, transfers, and result materialization should avoid global blocking.

3. **Placement is explicit but overridable**  
   User hints, model metadata, and policy engine collaborate.

4. **Profile-driven adaptation**  
   Runtime collects traces and adjusts placement/batching policies.

5. **Degradation over failure**  
   Unsupported or failing NPU ops transparently fallback to CPU paths when feasible.

---

## 4. Runtime Building Blocks

## 4.1 Frontend API Layer

Responsibilities:

- Submit jobs/graphs/pipelines.
- Declare QoS requirements (latency, throughput, priority).
- Provide optional placement hints and resource constraints.

Examples:

- `submit_graph(graph, qos, hints)`
- `submit_pipeline(stages, data_contracts)`

## 4.2 IR and Graph Planner

Responsibilities:

- Normalize user program into a runtime IR.
- Annotate operators with support matrix (`CPU`, `NPU`, both).
- Build cost model estimates (compute, transfer, memory).
- Partition graph into execution segments to reduce device boundaries.

Outputs:

- Physical execution plan with segment-to-device assignments.

## 4.3 Placement Policy Engine

Responsibilities:

- Decide initial placement using static heuristics + live telemetry.
- Adapt placement under load (hotspot migration, fallback, rebalance).

v1 strategy:

- Rule-based placement with per-op thresholds.
- Prefer NPU for supported high-arithmetic-intensity operations.
- Prefer CPU for control-heavy, tiny tensor, or unsupported operations.

## 4.4 Distributed Scheduler

Responsibilities:

- Place execution segments on node-level worker pools.
- Coordinate data locality and transfer-aware scheduling.
- Enforce QoS and fairness across jobs.

Core concepts:

- **Global queue** for incoming work.
- **Per-node local queues** with device-specific lanes (CPU lane, NPU lane).
- **Backpressure mechanism** to avoid queue explosion and OOM.

## 4.5 Execution Engine

Responsibilities:

- Execute DAG segments asynchronously.
- Manage dependencies through futures/events.
- Support micro-batching and dynamic batching where useful.

Worker model:

- CPU worker pool for orchestration and CPU kernels.
- NPU worker pool for accelerator kernels.
- Transfer workers for DMA/copy and format conversion.

## 4.6 Memory and Data Transfer Manager

Responsibilities:

- Unified tensor metadata and ownership tracking.
- Pinned CPU buffers and zero-copy where supported.
- Cross-device format conversion and layout management.
- Buffer reuse to reduce allocation churn.

## 4.7 Fault Tolerance and Fallback Manager

Responsibilities:

- Detect failed tasks/operators/nodes.
- Apply policy: retry, fallback (NPU -> CPU), or fail-fast.
- Maintain idempotent task semantics where applicable.

## 4.8 Observability and Control Plane

Responsibilities:

- Collect metrics, logs, traces.
- Expose runtime introspection APIs.
- Drive adaptive policy loops.

Key telemetry:

- Per-op latency by device.
- Queue wait time and utilization.
- Transfer time and bytes moved.
- Fallback counts and reasons.
- Tail latency (p95/p99) per workload class.

---


## 4.9 Lower-Level Runtime Interaction (Control/Data Plane Contract)

To ensure smooth interaction with lower-level runtimes/drivers (including Linqu runtime style integrations), PyPTO v1 should define a strict boundary contract rather than assuming direct kernel-level control.

### Control-plane contract

- **Lifecycle APIs:** `init_device`, `load_artifact`, `create_session`, `destroy_session`.
- **Execution APIs:** `enqueue(op_or_subgraph, inputs, attrs, stream_id)` + completion events.
- **Resource APIs:** query capabilities (op support, precision, memory limits, stream count).
- **Health APIs:** heartbeat, error classification, degraded-mode signals.

### Data-plane contract

- Common tensor descriptor (`shape`, `dtype`, `layout`, `strides`, `device`, `memory_kind`).
- Explicit ownership semantics (who allocates/frees, borrow vs transfer).
- Transfer primitives (`host<->npu`, `npu<->npu`) with async event handles.
- Optional zero-copy / shared-memory pathways behind capability flags.

### Compatibility and versioning

- Runtime contract versioning (`major.minor`) with feature negotiation at startup.
- Capability bitset for optional features (quantization modes, graph execution, fused ops).
- Backward-compatible fallback path when capability is missing.

### Error model alignment

- Canonical error classes: `RETRYABLE`, `FALLBACKABLE`, `FATAL`, `RESOURCE_EXHAUSTED`, `UNSUPPORTED`.
- Lower-level error normalization into canonical classes before scheduler action.

This contract-first approach allows PyPTO to integrate heterogeneous lower-level stacks consistently while keeping upper-layer scheduling logic portable.

## 5. Execution Model

## 5.1 Plan Lifecycle

1. Ingest program/job.
2. Convert to IR and annotate support/cost.
3. Partition and place segments.
4. Dispatch tasks to distributed scheduler.
5. Execute asynchronously with dependency tracking.
6. Continuously profile and adapt future placements.

## 5.2 Data and Control Flow

- **Data flow:** tensors/records through stage boundaries.
- **Control flow:** execution tokens/events for readiness and completion.
- **Policy flow:** telemetry feeds back into placement/scheduling decisions.

---

## 6. Scenario-to-Design Mapping

## 6.1 LLM Serving

- Dynamic batch assembler on CPU.
- NPU execution for transformer blocks.
- CPU sampling/post-processing path.
- KV-cache locality awareness in scheduler.

## 6.2 RL

- CPU actor pools for environment interaction.
- NPU learner pools for training updates.
- Replay buffer service with bounded-latency fetch.
- Async parameter broadcast and staleness controls.

## 6.3 Continuous Learning

- Dual-lane runtime: online inference lane + adaptation lane.
- Budgeted resource allocation to protect serving SLO.
- Trigger-based retraining schedule from drift signals.

## 6.4 Multimodal

- CPU preprocess stages per modality.
- NPU modality encoders and fusion models.
- Transfer manager aligns representation formats across stages.

## 6.5 Recommendation/Ranking

- CPU candidate generation and feature joining.
- NPU dense scoring/reranking.
- Adaptive batching based on request mix and SLA classes.

---

## 7. Resource Model

Runtime should represent resources explicitly:

- Node capacities: CPU cores, memory, NPU count, NPU memory, network bandwidth.
- Device capabilities: supported ops, precision modes, memory hierarchy.
- Workload class constraints: latency SLO, throughput target, priority.

This allows admission control and predictable multi-tenant behavior.

---

## 8. QoS and Scheduling Policies (v1)

1. **Priority-aware queues** (e.g., realtime > batch).
2. **SLO-aware dispatch** for low-latency classes.
3. **Work-conserving fallback** when NPUs are saturated.
4. **Quota enforcement** to prevent starvation.
5. **Adaptive micro-batch size** based on observed latency/throughput.

---

## 9. Failure Model

- **Operator-level failure:** retry or fallback to CPU.
- **Worker-level failure:** task reschedule on healthy worker.
- **Node-level failure:** reroute through scheduler with state recovery where possible.
- **Overload condition:** trigger backpressure + admission throttling.

Design preference in v1: preserve service continuity over peak efficiency.

---

## 10. Security and Multi-Tenancy Considerations

- Namespace-level isolation for jobs.
- Resource quotas per tenant.
- Signed model/artifact loading.
- Audit logs for placement and execution decisions.

---

## 11. v1 Implementation Roadmap

## Phase 1: Core Runtime Skeleton

- IR definition
- CPU/NPU worker abstraction
- Simple partitioner + rule-based placement
- Basic distributed scheduler and queueing
- Lower-level runtime adapter interface (v1 control/data plane contract)

## Phase 2: Production Baselines

- Transfer/memory manager
- Capability negotiation + error normalization with lower-level runtime
- Fallback/retry policy engine
- Metrics/tracing and dashboards
- Dynamic batching (LLM + ranking focus)

## Phase 3: Adaptive Intelligence

- Cost model calibration from telemetry
- Policy feedback loops for placement
- RL/continuous-learning-specific scheduling plugins

---

## 12. Open Questions

1. What is the primary v1 SLA target: low-latency serving or maximum throughput batch?
2. Which NPU backend(s) should be first-class initially?
3. How much user placement control should be exposed vs hidden?
4. Is interoperability with existing frameworks required at v1 (e.g., PyTorch graph import)?
5. What consistency guarantees are needed for stateful workloads (e.g., KV cache, replay buffers)?
6. Which lower-level runtime ABI should be treated as the reference adapter contract in v1?

---

## 13. Suggested v1 Scope Recommendation

If only one path can be prioritized, start with:

- **LLM serving + ranking inference** as first-class workloads,
- with robust CPU/NPU mixed scheduling,
- and reusable primitives (partition, async execution, transfer, fallback, telemetry).

This offers immediate practical value and establishes the architectural base for RL and continuous learning expansion.


---

## 14. Proposed Adapter Interface Sketch (for Lower-Level Integration)

```text
interface LowerRuntimeAdapter {
  Capabilities get_capabilities();
  Session create_session(ModelArtifact artifact, SessionConfig cfg);
  Event enqueue(Session session, ExecutableUnit unit, TensorRef[] inputs, AttrMap attrs, StreamId stream);
  Status poll(Event event);
  TensorRef materialize(Event event, OutputIndex idx);
  void release(TensorRef tensor);
  void destroy_session(Session session);
}
```

Notes:

- `ExecutableUnit` can be either single op or pre-partitioned subgraph.
- Event-driven polling/callback avoids global synchronization.
- Adapter hides backend-specific APIs while exposing capabilities for planner/scheduler decisions.
