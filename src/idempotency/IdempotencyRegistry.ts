import { IdempotencyPolicy } from '../policy/RuntimePolicy'
import { IdempotencyRecord, IdempotencyStatus } from '../shared/types'
import { computeIdempotencyKey } from './KeyComputer'
import { PayloadNormalizer } from './PayloadNormalizer'

export class IdempotencyRegistry {
  private readonly records = new Map<string, IdempotencyRecord>()
  private readonly normalizer: PayloadNormalizer

  constructor(private readonly policy: IdempotencyPolicy) {
    this.normalizer = new PayloadNormalizer(policy)
  }

  computeKey(input: {
    action: string
    payload?: Record<string, unknown>
    sessionId?: string
    taskId?: string
    scope?: 'session' | 'task' | 'global'
  }): string {
    const scope = input.scope ?? this.policy.getScopeForAction(input.action)
    const canonicalPayload = this.normalizer.normalize(input.payload)
    return computeIdempotencyKey({ action: input.action, scope, canonicalPayload })
  }

  check(key: string): { status: IdempotencyStatus; record?: IdempotencyRecord } {
    const record = this.records.get(key)
    if (!record) return { status: 'not_seen' }
    return { status: record.status, record }
  }

  start(key: string, action = 'unknown', scope: 'session' | 'task' | 'global' = 'session'): void {
    this.records.set(key, {
      key,
      action,
      scope,
      status: 'in_flight',
      startedAt: Date.now(),
    })
  }

  complete(key: string, result?: unknown, meta?: { prevented?: boolean; preventedReason?: string }): void {
    const existing = this.records.get(key)
    if (!existing) throw new Error(`Idempotency record not found for ${key}`)

    this.records.set(key, {
      ...existing,
      status: 'completed',
      completedAt: Date.now(),
      result,
      prevented: Boolean(meta?.prevented),
      preventedReason: meta?.preventedReason,
    })
  }

  fail(key: string, error?: string): void {
    const existing = this.records.get(key)
    if (!existing) throw new Error(`Idempotency record not found for ${key}`)

    this.records.set(key, {
      ...existing,
      status: 'failed',
      completedAt: Date.now(),
      error,
    })
  }

  get(key: string): IdempotencyRecord | undefined {
    return this.records.get(key)
  }
}
