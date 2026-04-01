import { IdempotencyPolicy } from '../policy/RuntimePolicy'
import { IdempotencyRecord, IdempotencyStatus } from '../shared/types'
import { KeyComputer } from './KeyComputer'

export class IdempotencyRegistry {
  private readonly records = new Map<string, IdempotencyRecord>()
  private readonly keyComputer: KeyComputer

  constructor(private readonly policy: IdempotencyPolicy) {
    this.keyComputer = new KeyComputer(policy)
  }

  computeKey(input: {
    action: string
    payload?: Record<string, unknown>
    sessionId?: string
    taskId?: string
    scope?: 'session' | 'task' | 'global'
  }): string {
    const scope = input.scope ?? this.policy.getScopeForAction(input.action)
    return this.keyComputer.compute({ action: input.action, scope, payload: input.payload })
  }

  check(key: string): { status: IdempotencyStatus; record?: IdempotencyRecord } {
    const record = this.records.get(key)
    if (!record) return { status: 'not_seen' }
    return { status: record.status, record }
  }

  checkAndClaim(input: { key: string; action: string; scope: 'session' | 'task' | 'global'; executionId: string }): {
    status: 'claimed' | 'in_flight' | 'completed' | 'failed'
    record?: IdempotencyRecord
  } {
    const existing = this.records.get(input.key)
    if (!existing) {
      const claimed: IdempotencyRecord = {
        key: input.key,
        action: input.action,
        scope: input.scope,
        status: 'in_flight',
        startedAt: Date.now(),
        executionId: input.executionId,
      }
      this.records.set(input.key, claimed)
      return { status: 'claimed', record: claimed }
    }

    if (existing.status === 'completed') return { status: 'completed', record: existing }
    if (existing.status === 'failed') return { status: 'failed', record: existing }
    return { status: 'in_flight', record: existing }
  }

  start(key: string, action: string, scope: 'session' | 'task' | 'global'): void {
    const existing = this.records.get(key)
    if (existing?.status === 'in_flight') {
      throw new Error(`Action already in flight for ${key}`)
    }
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
