import test from 'node:test'
import assert from 'node:assert/strict'

import { ExecutionRegistry } from '../src/execution/ExecutionRegistry.ts'
import { IdempotencyRegistry } from '../src/idempotency/IdempotencyRegistry.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'

test('idempotency check returns status and record', () => {
  const reg = new IdempotencyRegistry(defaultRuntimePolicy.idempotency)
  const key = reg.computeKey({ action: 'a', payload: { x: 1 } })
  assert.equal(reg.check(key).status, 'not_seen')
  reg.start(key, 'a', 'session')
  reg.complete(key, { done: true }, { prevented: true, preventedReason: 'blocked' })
  const check = reg.check(key)
  assert.equal(check.status, 'completed')
  assert.equal(check.record?.prevented, true)
  assert.equal(check.record?.preventedReason, 'blocked')
})

test('checkAndClaim is atomic for repeated claims', () => {
  const reg = new IdempotencyRegistry(defaultRuntimePolicy.idempotency)
  const key = reg.computeKey({ action: 'a', payload: { x: 1 } })

  const first = reg.checkAndClaim({ key, action: 'a', scope: 'session', executionId: 'e1' })
  const second = reg.checkAndClaim({ key, action: 'a', scope: 'session', executionId: 'e2' })

  assert.equal(first.status, 'claimed')
  assert.equal(second.status, 'in_flight')
})

test('replay path blocks embedding lookup and enforces version pinning', () => {
  const ex = new ExecutionRegistry()
  ex.append({
    executionId: 'e1',
    sessionId: 's1',
    idempotencyKey: 'k',
    determinism: { level: 'deterministic' },
    replayContext: { operatorVersion: '1', schemaVersion: '1', runtimeVersion: '0.4.0' },
    input: { raw: 'x', domain: 'ops' },
    routing: { source: 'db_execute', operator: 'op1' },
    action: { name: 'act' },
    result: { success: true },
    outcome: { trigger: 't1', walPatternClass: 'w1' },
  })

  assert.equal(
    ex.getReplayCandidates('act', {
      expectedContext: { operatorVersion: '1', schemaVersion: '1', runtimeVersion: '0.4.0' },
    }).length,
    1,
  )

  assert.equal(
    ex.getReplayCandidates('act', {
      expectedContext: { operatorVersion: '2', schemaVersion: '1', runtimeVersion: '0.4.0' },
    }).length,
    0,
  )

  assert.throws(() => ex.getReplayCandidates('act', { embeddingQuery: 'vector' }), /forbidden/)
})
