import { CostBreakdown, DecisionSource, ExecutionRecord } from '../shared/types'

export type MetricEvent = {
  type: 'execution_complete' | 'idempotency_hit' | 'in_flight_collision'
  timestamp: number
  payload: unknown
}

export type RuntimeMetricsSnapshot = {
  totalExecutions: number
  deterministicRate: number
  boundedRate: number
  unrestrictedRate: number
  fallbackReasons: Record<string, number>
  idempotency: {
    hitRate: number
    preventedRate: number
    inFlightCollisions: number
  }
  latency: {
    p50: number
    p95: number
    p99: number
  }
  costs: CostBreakdown
}

export class RuntimeMetrics {
  private readonly executions: ExecutionRecord[] = []
  private idempotencyHits = 0
  private inFlightCollisions = 0
  private readonly listeners = new Set<(event: MetricEvent) => void>()

  subscribe(listener: (event: MetricEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  recordExecution(record: ExecutionRecord): void {
    this.executions.push(record)
    this.emit({ type: 'execution_complete', timestamp: Date.now(), payload: record })
  }

  recordIdempotencyHit(): void {
    this.idempotencyHits += 1
    this.emit({ type: 'idempotency_hit', timestamp: Date.now(), payload: { hits: this.idempotencyHits } })
  }

  recordInFlightCollision(): void {
    this.inFlightCollisions += 1
    this.emit({ type: 'in_flight_collision', timestamp: Date.now(), payload: { collisions: this.inFlightCollisions } })
  }

  snapshot(): RuntimeMetricsSnapshot {
    const total = this.executions.length
    const bySource = this.countBySource()
    const fallbackReasons: Record<string, number> = {}
    const latencyValues = this.executions.map((e) => e.durationMs ?? 0).sort((a, b) => a - b)

    let prevented = 0
    const costs: CostBreakdown = { deterministic: 0, bounded: 0, unrestricted: 0, total: 0 }

    for (const ex of this.executions) {
      if (ex.result.prevented) prevented += 1
      if (ex.routing.fallbackReason) fallbackReasons[ex.routing.fallbackReason] = (fallbackReasons[ex.routing.fallbackReason] ?? 0) + 1
      costs.deterministic += ex.cost.deterministic
      costs.bounded += ex.cost.bounded
      costs.unrestricted += ex.cost.unrestricted
      costs.total += ex.cost.total
    }

    return {
      totalExecutions: total,
      deterministicRate: total ? ((bySource.deterministic ?? 0) / total) : 0,
      boundedRate: total ? ((bySource.bounded ?? 0) / total) : 0,
      unrestrictedRate: total ? ((bySource.unrestricted ?? 0) / total) : 0,
      fallbackReasons,
      idempotency: {
        hitRate: total ? this.idempotencyHits / total : 0,
        preventedRate: total ? prevented / total : 0,
        inFlightCollisions: this.inFlightCollisions,
      },
      latency: {
        p50: percentile(latencyValues, 50),
        p95: percentile(latencyValues, 95),
        p99: percentile(latencyValues, 99),
      },
      costs,
    }
  }

  private emit(event: MetricEvent): void {
    for (const listener of this.listeners) listener(event)
  }

  private countBySource(): Record<'deterministic' | 'bounded' | 'unrestricted', number> {
    const out = { deterministic: 0, bounded: 0, unrestricted: 0 }
    for (const ex of this.executions) {
      if (ex.determinism.level === 'deterministic') out.deterministic += 1
      if (ex.determinism.level === 'bounded') out.bounded += 1
      if (ex.determinism.level === 'unrestricted') out.unrestricted += 1
    }
    return out
  }
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0
  const idx = Math.ceil((p / 100) * values.length) - 1
  return values[Math.max(0, idx)]
}

export function estimateCost(source: DecisionSource): CostBreakdown {
  if (source === 'turbo_assist') return { deterministic: 0, bounded: 0.001, unrestricted: 0, total: 0.001 }
  if (source === 'llm_freeform') return { deterministic: 0, bounded: 0, unrestricted: 0.01, total: 0.01 }
  if (source === 'llm_force_freeform') return { deterministic: 0, bounded: 0, unrestricted: 0.02, total: 0.02 }
  return { deterministic: 0, bounded: 0, unrestricted: 0, total: 0 }
}
