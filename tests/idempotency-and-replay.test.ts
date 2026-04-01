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

test('replay path blocks embedding lookup', () => {
  const ex = new ExecutionRegistry()
  ex.append({
    sessionId: 's1',
    idempotencyKey: 'k',
    input: { raw: 'x', domain: 'ops' },
    routing: { source: 'db_execute', operator: 'op1' },
    action: { name: 'act' },
    result: { success: true },
    outcome: { trigger: 't1', walPatternClass: 'w1' },
  })

  assert.equal(ex.getReplayCandidates('act').length, 1)
  assert.throws(() => ex.getReplayCandidates('act', { embeddingQuery: 'vector' }), /forbidden/)
})
