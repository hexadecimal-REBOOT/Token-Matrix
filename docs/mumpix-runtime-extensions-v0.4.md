# Mumpix Runtime Extensions

> **Formal Technical Specification v0.4 — Pre-build Final**

A Mumpix-native runtime architecture for deterministic agent execution, auditable background behavior, and idempotent action dispatch.

-----

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [System Invariants](#system-invariants)
- [Architecture](#architecture)
- [Module Layout](#module-layout)
- [Core Data Types](#core-data-types)
- [Canonical Payload Normalization](#canonical-payload-normalization)
- [Task Registry](#task-registry)
- [Dream Engine](#dream-engine)
- [Execution Registry](#execution-registry)
- [Idempotency Registry](#idempotency-registry)
- [Runtime Core](#runtime-core)
- [Policy Layer](#policy-layer)
- [Invariant Reference](#invariant-reference)
- [Metrics](#metrics)
- [End-to-End Example](#end-to-end-example)
- [Build Checklist](#build-checklist)
- [What This Spec Prevents](#what-this-spec-prevents)

-----

## Overview

This specification defines six foundational runtime systems for Mumpix:

|System                  |Purpose                                                                                                                 |
|------------------------|------------------------------------------------------------------------------------------------------------------------|
|**Task Registry**       |Makes all background and long-running work visible, controllable, and auditable                                         |
|**Dream Engine**        |Bounded, deterministic memory consolidation and gene promotion in the background                                        |
|**Execution Registry**  |Complete record of every action, decision origin, outcome, and replay signal                                            |
|**Idempotency Registry**|Prevents duplicate execution across replay, retry, and concurrent paths                                                 |
|**Runtime Core**        |Separates engine, runtime, and transport for consistent execution across CLI, MCP, server, and orchestrated environments|
|**Policy Layer**        |Explicit home for all runtime policy decisions — no policy hardcoded in runtime files                                   |

These systems extend the Mumpix family without introducing transcript-heavy, retrieval-first patterns.

-----

## Design Principles

**1. Local-first**
All core behavior must work without network access.

**2. Deterministic where possible**
Structured truth, execution routing, promotion, and enforcement must not depend on probabilistic re-interpretation once sufficient evidence exists.

**3. Background work must be inspectable**
No invisible autonomous behavior. Every background process must appear in the task system.

**4. Memory is split by role**

- `Turbo` = working memory
- `MumpixDB` = structured truth
- `MumpixDNA` = promoted behavior

**5. Promotion over summarization**
Repeated useful patterns become behavior — not permanent text bloat.

**6. Exceptions must be audited**
Every bypass of a deterministic path is an exception. Exceptions must be declared, logged, and visible in coverage views.

**7. Probabilistic fallback is tiered**
After deterministic paths are exhausted, the runtime may enter **bounded probabilistic fallback** (`turbo_assist`) before **unrestricted probabilistic fallback** (`llm_freeform` / `llm_force_freeform`). These are distinct tiers, not a single fallback category.

-----

## System Invariants

> **These invariants are enforced in RuntimeCore. They are not conventions. No subsystem may bypass them.**

-----

### Invariant 1 — Deterministic-First Routing

No input may skip to unrestricted probabilistic fallback while a deterministic path remains unresolved.

```text
Deterministic tier
  1. DNA BLOCK
  2. DNA CHECK
  3. DNA DO
  4. DB executable operator

Bounded probabilistic tier
  5. Turbo-assisted LLM

Unrestricted probabilistic tier
  6a. Freeform LLM       — all deterministic + bounded paths exhausted
  6b. Force-freeform LLM — explicit exception; valid forceFreeform declaration required
```

A `force_freeform` declaration missing required fields must be **rejected** — not silently downgraded to normal fallback.

**Enforcement point:** `RuntimeCore`, not `DecisionRouter` convention.

-----

### Invariant 2 — Dream Safety

Dream must never mutate genes currently active in a running session.

Promotion and decay writes are deferred until the affected genes are no longer session-bound. Deferred genes accumulate metadata. Starvation prevention is mandatory — genes may not be deferred indefinitely.

-----

### Invariant 3 — Replay Determinism

Replay candidate lookup must use structural similarity only.

Semantic or embedding-based similarity is **forbidden** for deterministic replay selection.

Structural match order:

1. Exact action sequence match
2. Exact operator sequence match
3. Exact domain + trigger + outcome signature
4. Exact WAL sequence pattern class

-----

### Invariant 4 — PINNED Firing Threshold

PINNED protects gene survival only. A pinned gene may fire only when **both** conditions are satisfied:

- Normal context match passes
- Match confidence ≥ `PINNED_FIRE_THRESHOLD`

`PINNED_FIRE_THRESHOLD` must be **stricter** than the threshold for ordinary genes.

```typescript
if (!matched) continue
if (gene.pinned && matchConfidence < PINNED_FIRE_THRESHOLD) continue
```

-----

### Invariant 5 — Idempotency

No action may execute twice with the same normalized payload in the same scope once a successful completion record exists, unless explicitly marked `re_executable: true`.

**Prevented-action rule:** A prevented action (BLOCK fire) is recorded as a completed idempotency resolution. Prevention is the definitive terminal result. The system will not retry a prevented action in the same scope unless the prevention condition changes. `IdempotencyRecord.prevented: true` distinguishes prevention completions from execution completions.

-----

### Invariant 6 — No Promotion Without Proof

Genes may not be promoted without satisfying the evidence threshold. Dream may not promote based on single observations.

-----

### Invariant 7 — Shared Truth, Local Working Memory

DB and DNA may be shared across agents and sessions. Turbo is local to the agent/session. No agent may read another agent’s Turbo directly.

-----

### Invariant 8 — Background Tasks Must Be Visible

All background work must be registered in `TaskRegistry` before work begins. Tasks that terminate without reaching a terminal state (`completed` / `failed` / `killed`) are a spec violation.

-----

## Architecture

```text
Input
→ Runtime Core
  → Policy check          (RoutingPolicy)
  → Idempotency check     (Invariant 5)
  → Decision Router       (Invariant 1 enforced)

    ── Deterministic tier ──────────────────────
    → MumpixDNA   BLOCK → CHECK → DO
    → MumpixDB    operator execution

    ── Bounded probabilistic tier ──────────────
    → Turbo-assisted LLM

    ── Unrestricted probabilistic tier ─────────
    → LLM freeform          (normal exhaustion)
    → LLM freeform          (force exception — audited)

  → Execution Registry
  → Task Registry
  → Dream Engine            (background, Invariant 2 enforced)
```

-----

## Module Layout

```text
src/
  runtime/
    RuntimeCore.ts
    DecisionRouter.ts
    SessionRuntime.ts
    TransportAdapter.ts
    ContextAssembler.ts

  policy/
    RuntimePolicy.ts
    RoutingPolicy.ts
    IdempotencyPolicy.ts
    DreamPolicy.ts

  tasks/
    Task.ts
    TaskRegistry.ts
    TaskFramework.ts
    TaskTypes.ts
    tasks/
      DreamTask.ts
      ReplayTask.ts
      PromotionSweepTask.ts
      FleetPatchTask.ts
      SubAgentTask.ts

  dream/
    DreamEngine.ts
    DreamPlanner.ts
    DreamSweep.ts
    DreamPruner.ts
    DreamPromoter.ts
    DreamDecay.ts
    DreamDeferralTracker.ts

  execution/
    ExecutionRegistry.ts
    ExecutionRecord.ts
    OutcomeTracker.ts
    ReplayStore.ts
    CoverageTracker.ts
    DecisionSourceTracker.ts

  idempotency/
    IdempotencyRegistry.ts
    IdempotencyRecord.ts
    KeyComputer.ts
    PayloadNormalizer.ts

  orchestration/
    Orchestrator.ts
    SubAgentSpawner.ts
    DomainFilter.ts
    OutcomeCollector.ts
    FleetPatch.ts

  turbo/
    TurboMemory.ts
    TurboRecall.ts
    TurboPrune.ts

  db/
    MumpixDbAdapter.ts
    StateStore.ts
    TimelineStore.ts
    FactStore.ts
    OutcomeStore.ts

  dna/
    DNAStrand.ts
    DNARuntime.ts
    PromotionEngine.ts
    DecayEngine.ts
    DriftDetector.ts
    GeneMatcher.ts

  shared/
    types.ts
    ids.ts
    errors.ts
```

-----

## Core Data Types

### TaskStateBase

```typescript
type TaskStateBase = {
  id: string
  type: string
  title: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'killed'
  createdAt: number
  startTime?: number
  endTime?: number
  notified: boolean
}
```

### DecisionSource

```typescript
type DecisionSource =
  | 'dna_block'
  | 'dna_check'
  | 'dna_do'
  | 'db_execute'
  | 'turbo_assist'
  | 'llm_freeform'
  | 'llm_force_freeform'   // unrestricted tier, explicit exception — tracked separately
  | 'replay'
```

### CheckOutcome

```typescript
type CheckOutcome =
  | 'check_pass'            // condition satisfied — proceed
  | 'check_fail'            // terminal failure under current state; do not retry without state change
  | 'prerequisite_required' // recoverable — missing precondition may be resolved before retry
```

> `check_fail` means the system evaluated the condition and the answer is definitively no.
> `prerequisite_required` means the condition cannot yet be evaluated — resolution is possible before retry.

### ExecutionRecord

```typescript
type ExecutionRecord = {
  id: string
  sessionId: string
  taskId?: string
  timestamp: number
  idempotencyKey: string

  input: {
    raw: string
    parsedIntent?: string
    domain?: string
  }

  routing: {
    source: DecisionSource
    matchedGeneId?: string
    checkOutcome?: CheckOutcome
    operator?: string
    fallbackReason?: string
    forceFreeform?: {
      reason: string
      callerSource: string
      approvedAt: number      // enables policy audit trail; required when present
    }
  }

  action: {
    name: string
    args?: Record<string, unknown>
    reExecutable?: boolean
  }

  result: {
    success: boolean
    output?: unknown
    error?: string
    prevented?: boolean
    shortCircuited?: boolean
  }

  outcome: {
    reward?: number
    damage?: number
    tags?: string[]
  }
}
```

### DreamTaskState

```typescript
type DreamTaskState = TaskStateBase & {
  type: 'dream'
  phase: 'starting' | 'sweeping' | 'promoting' | 'decaying' | 'pruning' | 'completing'
  sessionsReviewing: number
  filesTouched: string[]
  genesDeferred: string[]
  genesStarving: string[]
  turns: Array<{ text: string; toolUseCount: number }>
  abortController?: AbortController
  priorMtime: number
}
```

### IdempotencyRecord

```typescript
type IdempotencyStatus = 'not_seen' | 'in_flight' | 'completed' | 'failed'

type IdempotencyRecord = {
  key: string
  action: string
  scope: 'session' | 'task' | 'global'
  status: IdempotencyStatus
  startedAt?: number
  completedAt?: number
  result?: unknown
  error?: string
  prevented?: boolean         // true when completion was a BLOCK fire, not an execution
  preventedReason?: string    // optional — for cleaner short-circuit audit logs
}
```

### DeferralRecord

```typescript
type DeferralRecord = {
  geneId: string
  deferredCount: number
  firstDeferredAt: number
  lastDeferredAt: number
  starving: boolean
}
```

-----

## Canonical Payload Normalization

Used by `IdempotencyRegistry` to compute deterministic keys. All implementations must follow these rules exactly. Divergence produces key mismatches across runtimes.

### Normalization Rules

|Rule              |Behavior                                                                           |
|------------------|-----------------------------------------------------------------------------------|
|Object keys       |Sorted recursively — all levels                                                    |
|Arrays            |Preserved in original order                                                        |
|`undefined` fields|Dropped entirely                                                                   |
|`null`            |Retained as-is                                                                     |
|Numbers           |Canonical serialization — `1` and `1.0` produce identical output                   |
|`NaN` / `Infinity`|Rejected unless field declares `allow_special_numeric: true` in `IdempotencyPolicy`|
|Booleans          |Preserved as-is                                                                    |
|Strings           |Preserved exactly — no trimming, no case normalization                             |
|Timestamps        |Excluded by default unless declared as semantic fields in `IdempotencyPolicy`      |
|Transient fields  |Excluded by policy declaration in `IdempotencyPolicy`                              |
|Hash input        |Canonical JSON string, UTF-8 encoded                                               |

### Key Structure

```text
{action_name}:{scope}:{sha256_of_canonical_json}
```

### Semantic Field Declaration

```typescript
// IdempotencyPolicy
semanticTimestampFields: ['scheduled_for', 'effective_date']
```

Undeclared time-related fields are excluded from hash input.

### Transient Field Exclusion

```typescript
// IdempotencyPolicy
transientFields: ['request_id', 'trace_id', 'correlation_id', 'client_timestamp']
```

-----

## Task Registry

### Purpose

Authoritative system for all background and long-running work. All background jobs must call `registerTask()` before work starts (Invariant 8).

### Supported Task Types

|Type             |Description                                         |
|-----------------|----------------------------------------------------|
|`dream`          |Memory consolidation and gene promotion cycle       |
|`promotion_sweep`|Standalone promotion pass over candidates           |
|`replay`         |Build or validate replay chains                     |
|`fleet_patch`    |Apply CRISPR patch to master + all active sub-agents|
|`sub_agent`      |Long-running domain-specialized sub-agent job       |

### Interface

```typescript
interface TaskRegistry {
  register(task: TaskStateBase): string
  update<T extends TaskStateBase>(taskId: string, updater: (task: T) => T): void
  complete(taskId: string): void
  fail(taskId: string, error?: string): void
  kill(taskId: string): Promise<void>
  list(): TaskStateBase[]
  get(taskId: string): TaskStateBase | undefined
}
```

-----

## Dream Engine

### Purpose

Bounded consolidation over existing memory and behavior signals. Not a freeform summarizer. Invariant 2 applies to all writes.

### Session-Lock Enforcement

Before any promotion or decay write:

1. Resolve candidate gene IDs
2. Call `SessionRuntime.activeGeneSet()`
3. If any target gene is active → record in `DeferralTracker`, requeue, skip write
4. Only write when no target gene is active in any session

### Starvation Prevention

Genes that are repeatedly deferred accumulate a `DeferralRecord`. Policy:

```typescript
// DreamPolicy
maxDeferralCount: number              // emit operator alert when exceeded
maintenanceWindowFallback: boolean    // allow write during declared maintenance window
starvingGeneAlertThreshold: number    // secondary escalation threshold
maintenanceWindowActive(): boolean
```

When `deferredCount >= maxDeferralCount`:

- Gene added to `genesStarving` in `DreamTaskState`
- Operator alert emitted
- If `maintenanceWindowFallback: true` and window is active → write proceeds

### Dream Cycle

```text
Orientation
→ Gather Signal
→ Resolve session gene locks        (Invariant 2)
→ Consolidate
→ Promote                           (session-safe genes only)
→ Decay                             (session-safe genes only)
→ Prune
→ Update DeferralTracker
→ Check starvation thresholds       → emit alerts if any exceed
→ Requeue deferred genes
→ Emit task completion              (genesDeferred + genesStarving counts)
```

### Interface

```typescript
interface DreamEngine {
  start(opts: {
    sessionIds?: string[]
    abortController: AbortController
    priorMtime: number
  }): Promise<string>

  sweep(): Promise<DreamSweepResult>
  promote(): Promise<PromotionResult>
  decay(): Promise<DecayResult>
  prune(): Promise<PruneResult>
  getDeferred(): DeferralRecord[]
  getStarving(): DeferralRecord[]
}
```

-----

## Execution Registry

### Purpose

Complete record of what the system did, why it chose that path, and what happened. The missing layer between raw logs and agent memory.

### Coverage Tracking

|Metric                   |Notes                                                      |
|-------------------------|-----------------------------------------------------------|
|`% dna_block`            |                                                           |
|`% dna_do`               |                                                           |
|`% db_execute`           |                                                           |
|`% turbo_assist`         |Bounded probabilistic tier                                 |
|`% llm_freeform`         |Unrestricted — exhaustion path                             |
|`% llm_force_freeform`   |Unrestricted — exception path; **should trend toward zero**|
|CheckOutcome distribution|`check_pass` / `check_fail` / `prerequisite_required`      |
|Fallback rate            |Bounded and unrestricted reported separately               |

### Replay Support

Structural match only (Invariant 3). No embedding-based lookup in this path.

### Interface

```typescript
interface ExecutionRegistry {
  append(record: ExecutionRecord): string
  get(id: string): ExecutionRecord | undefined
  listBySession(sessionId: string): ExecutionRecord[]
  listByTask(taskId: string): ExecutionRecord[]
  listBySource(source: DecisionSource): ExecutionRecord[]
  listByCheckOutcome(outcome: CheckOutcome): ExecutionRecord[]
  getCoverageStats(): CoverageStats
  getReplayCandidates(action: string): ReplayCandidate[]
}
```

-----

## Idempotency Registry

### Purpose

Prevents duplicate execution across replay, retry, and concurrent paths. Required for correctness in any system supporting background tasks, replays, or fleet operations.

### force_freeform Policy

`force_freeform` is an exception path, not a convenience path.

A valid declaration requires:

- `forceFreeformReason` — non-empty string
- `callerSource` — non-empty string
- `approvedAt` — timestamp

Additional rules:

- May be denied by `RoutingPolicy` per domain
- Must be logged as an override event
- Appears in coverage under `llm_force_freeform` — separate from normal fallback
- Missing required fields → **request rejected**, not downgraded

```typescript
type RuntimeOptions = {
  force_freeform?: boolean
  forceFreeformReason?: string    // required when force_freeform is true
  callerSource?: string           // required when force_freeform is true
  sessionId?: string
  taskId?: string
  scope?: 'session' | 'task' | 'global'
}
```

### Routing Insertion Point

```text
route selected
→ IdempotencyRegistry.check(key)
  → { status: 'not_seen' }                  → proceed, .start(key)
  → { status: 'in_flight' }                 → surface conflict, wait or reject
  → { status: 'completed', record.prevented } → short-circuit, return prevention result
  → { status: 'completed' }                 → short-circuit, return record.result
  → { status: 'failed' }                    → surface error, allow retry per scope policy
→ execute action
→ .complete(key, result, { prevented: false }) or .fail(key, error)
→ ExecutionRegistry.append(record)
```

### Interface

```typescript
interface IdempotencyRegistry {
  computeKey(input: {
    action: string
    payload?: Record<string, unknown>
    sessionId?: string
    taskId?: string
    scope?: 'session' | 'task' | 'global'
  }): string

  check(key: string): {
    status: IdempotencyStatus
    record?: IdempotencyRecord
  }

  start(key: string): void
  complete(key: string, result?: unknown, meta?: { prevented?: boolean }): void
  fail(key: string, error?: string): void
  get(key: string): IdempotencyRecord | undefined
}
```

-----

## Runtime Core

### Subsystems

|Subsystem         |Role                                                           |
|------------------|---------------------------------------------------------------|
|`RuntimeCore`     |End-to-end execution lifecycle, invariant enforcement          |
|`DecisionRouter`  |Routes requests per canonical tier order (Invariant 1 enforced)|
|`SessionRuntime`  |Session state, history, background tasks, active gene set      |
|`TransportAdapter`|Maps runtime calls into stdio, HTTP, or in-process APIs        |
|`ContextAssembler`|Builds minimal input context for LLM fallback                  |

### Decision Routing

Three tiers. Each must be exhausted before descending.

**Deterministic tier — no probabilistic element:**

|Step|Action                                                                                                              |
|----|--------------------------------------------------------------------------------------------------------------------|
|1   |DNA BLOCK — if matched + confidence ≥ threshold → block, record, return                                             |
|2   |DNA CHECK — evaluate, return `CheckOutcome`; `check_fail` halts; `prerequisite_required` may trigger resolution pass|
|3   |DNA DO — if matched + confidence ≥ threshold → execute gene behavior                                                |
|4   |DB operator — if query is computable → execute, return structured result                                            |

**Bounded probabilistic tier:**

|Step|Action                                                                        |
|----|------------------------------------------------------------------------------|
|5   |Turbo-assist — context-window-bounded LLM; Turbo injects relevant surface only|

**Unrestricted probabilistic tier:**

|Step|Action                                                         |
|----|---------------------------------------------------------------|
|6a  |Freeform LLM — all deterministic + bounded paths exhausted     |
|6b  |Force-freeform LLM — valid `forceFreeform` declaration required|

PINNED gene check applies at steps 1–3 (Invariant 4).

### Interface

```typescript
interface RuntimeCore {
  handleInput(input: string, opts?: RuntimeOptions): Promise<RuntimeResult>
  replay(action: string): Promise<ReplayResult>
  startBackgroundTask(type: string, payload?: unknown): Promise<string>
  getSession(sessionId: string): SessionState | undefined
}
```

-----

## Policy Layer

Named, explicit home for all runtime policy decisions. No policy rule may be hardcoded inside a runtime file.

### RoutingPolicy

Controls which domains may use `force_freeform`, which routing steps may be skipped, and which domains require stricter deterministic enforcement.

```typescript
interface RoutingPolicy {
  allowForceFreeform(domain: string, callerSource: string): boolean
  requireStrictDeterminism(domain: string): boolean
  getRoutingOverrides(domain: string): Partial<RoutingOptions>
}
```

### IdempotencyPolicy

Declares semantic timestamp fields, transient fields, and scope rules per action type.

```typescript
interface IdempotencyPolicy {
  semanticTimestampFields: string[]
  transientFields: string[]
  getScopeForAction(action: string): 'session' | 'task' | 'global'
  isReExecutable(action: string): boolean
}
```

### DreamPolicy

Controls starvation thresholds, maintenance windows, and which gene operations/domains Dream may touch.

```typescript
interface DreamPolicy {
  maxDeferralCount: number
  maintenanceWindowFallback: boolean
  starvingGeneAlertThreshold: number
  allowedMutationOperations: Array<'promotion' | 'decay' | 'prune'>
  allowedGeneDomains?: string[]       // optional domain filter for Dream mutations
  maintenanceWindowActive(): boolean
}
```

### RuntimePolicy

Top-level policy resolution surface for RuntimeCore.

```typescript
interface RuntimePolicy {
  routing: RoutingPolicy
  idempotency: IdempotencyPolicy
  dream: DreamPolicy
}
```

-----

## Invariant Reference

|#|Name                              |Rule                                                                                                                         |
|-|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
|1|Deterministic-first routing       |No input skips to unrestricted probabilistic fallback while a deterministic path remains unresolved                          |
|2|Dream safety                      |Dream may not mutate genes active in a running session; starvation prevention mandatory                                      |
|3|Replay determinism                |Replay candidate lookup uses structural similarity only; no embeddings                                                       |
|4|PINNED firing threshold           |PINNED protects survival only; pinned genes must satisfy relevance threshold to fire                                         |
|5|Idempotency                       |No action executes twice with same normalized payload in same scope once completed; prevented actions complete the resolution|
|6|No promotion without proof        |Genes may not be promoted without satisfying the evidence threshold                                                          |
|7|Shared truth, local working memory|DB and DNA may be shared; Turbo is local to agent/session                                                                    |
|8|Background tasks must be visible  |All background work registered in TaskRegistry before work begins                                                            |

-----

## Metrics

### Task

- Running / failed / killed task counts
- Average task duration
- Deferred gene count per Dream cycle
- Starving gene count

### Runtime

- LLM calls total
- Bounded fallback rate (`turbo_assist`)
- Unrestricted fallback rate (`llm_freeform`)
- `llm_force_freeform` rate *(must trend toward zero)*
- DNA fires by type (BLOCK / CHECK / DO)
- DB executions
- CheckOutcome distribution

### Promotion

- Promoted / rejected / decayed genes
- Deferred promotions, drift flags

### Replay

- Hits / misses, path compression delta

### Safety

- Prevented actions, BLOCK fires, CHECK failures
- Idempotency short-circuits (prevented vs executed, reported separately)

-----

## End-to-End Example

**Input: `"Deploy this fix"`**

```text
RuntimeCore.handleInput("Deploy this fix", { sessionId: "s1" })

→ RoutingPolicy: domain = 'deploy' — no force_freeform declared
→ IdempotencyRegistry.check(key) → { status: 'not_seen' } → .start(key)

→ DecisionRouter — Deterministic tier
  → DNA CHECK: tests_pass?
      → DB PU-2: tests_pass = false
      → CheckOutcome: 'check_fail'
  → DNA BLOCK: deploy_on_check_fail fires
      → action prevented

→ ExecutionRegistry.append({
    source: 'dna_block',
    checkOutcome: 'check_fail',
    result: { prevented: true }
  })
→ IdempotencyRegistry.complete(key, undefined, { prevented: true })
   // record.prevented = true — resolution recorded, not retried in scope

→ Turbo stores failed deploy attempt

// Dream Engine — background sweep
→ DreamEngine.sweep()
→ Sees repeated deploy_fail cluster in ExecutionRegistry
→ SessionRuntime.activeGeneSet() → deploy_guard gene not active  [Invariant 2]
→ DreamPolicy.allowedMutationOperations includes 'promotion'
→ DreamPolicy.allowedGeneDomains includes 'deploy'
→ Promotion write proceeds
→ DNA promotion candidate confidence increases
```

-----

## Build Checklist

> Freeze spec. Allow clarifying edits only if implementation reveals something genuinely missing.

### Phase 1 — Core correctness foundations

- [ ] `PayloadNormalizer` — implement canonical normalization (section above)
- [ ] `PayloadNormalizer` — unit tests covering every normalization rule
- [ ] `IdempotencyRegistry` — key computation, check, start, complete, fail
- [ ] `IdempotencyRegistry` — prevented-action semantics (`meta.prevented`)
- [ ] `IdempotencyRegistry` — `check()` returns `{ status, record? }` — not bare status
- [ ] `ExecutionRegistry` — append, list, coverage stats
- [ ] `ExecutionRegistry` — `CheckOutcome` and `forceFreeform` fields wired in
- [ ] `RuntimeCore` — three-tier routing structure
- [ ] `RuntimeCore` — Invariant 1 enforced; no path to step 6 without exhausting 1–4
- [ ] `RuntimeCore` — `force_freeform` validation; reject if missing required fields
- [ ] Routing invariant tests — coverage across all 6 routing steps
- [ ] `TaskRegistry` — register, update, complete, fail, kill
- [ ] Policy stubs — `RoutingPolicy`, `IdempotencyPolicy`, `DreamPolicy`, `RuntimePolicy`
- [ ] Policy stubs wired into RuntimeCore

### Phase 2 — Memory consolidation and coverage visibility

- [ ] `DreamEngine` — sweep, promote, decay, prune
- [ ] `DreamDeferralTracker` — deferredCount, firstDeferredAt, lastDeferredAt
- [ ] `DreamEngine` — starvation prevention; operator alert when threshold exceeded
- [ ] `DreamEngine` — Invariant 2 session-lock check before every write
- [ ] Promotion sweep task
- [ ] Coverage instrumentation — bounded vs unrestricted fallback tracked separately
- [ ] PINNED threshold enforcement — Invariant 4
- [ ] `CheckOutcome` states wired into routing, metrics, and orchestration responses

### Phase 3 — Scale and distribution

- [ ] Replay engine — structural match only (Invariant 3)
- [ ] Replay engine — embedding lookup explicitly blocked in this path
- [ ] Sub-agent spawning — domain-filtered, shared DB/DNA, local Turbo
- [ ] Fleet patching — CRISPR patch propagation master → descendants
- [ ] Orchestrator — outcome collection, promotion flow upward

-----

## What This Spec Prevents

|Risk                                                 |Prevention                                                                          |
|-----------------------------------------------------|------------------------------------------------------------------------------------|
|LLM shortcut becoming the default path               |Invariant 1 + three-tier model + `force_freeform` audit trail                       |
|Dream silently breaking live agent behavior          |Invariant 2 + session-lock check before every write                                 |
|Genes deferred forever without visibility            |Starvation prevention + `DeferralRecord` + operator alerts                          |
|Replay introducing non-determinism                   |Invariant 3 + structural-match-only rule                                            |
|Pinned genes becoming unconditional overrides        |Invariant 4 + `PINNED_FIRE_THRESHOLD`                                               |
|Duplicate execution in replay/retry paths            |Invariant 5 + `IdempotencyRegistry`                                                 |
|Ambiguity between blocked and executed completions   |`IdempotencyRecord.prevented` + prevented-action rule                               |
|Ambiguous CHECK results causing routing inconsistency|Typed `CheckOutcome` with explicit `check_fail` vs `prerequisite_required` semantics|
|Policy logic scattered across runtime files          |Named policy layer — four interfaces, one directory                                 |
|Idempotency key divergence across implementations    |Canonical normalization spec                                                        |
|Bounded vs unrestricted fallback conflated in metrics|Separate `DecisionSource` values + separate coverage tracking                       |

-----

*MumpixDB · VDSX Cloud · Runtime Extensions v0.4 · 2026*
