import test from 'node:test'
import assert from 'node:assert/strict'

import { MumpixDbAdapter } from '../src/db/MumpixDbAdapter.ts'
import { DNARuntime } from '../src/dna/DNARuntime.ts'
import { ExecutionRegistry } from '../src/execution/ExecutionRegistry.ts'
import { IdempotencyRegistry } from '../src/idempotency/IdempotencyRegistry.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'
import { RuntimeCore } from '../src/runtime/RuntimeCore.ts'
import { TaskRegistry } from '../src/tasks/TaskRegistry.ts'

function buildRuntime(policyOverride?: Partial<typeof defaultRuntimePolicy>) {
  const policy = { ...defaultRuntimePolicy, ...policyOverride }
  const dna = new DNARuntime([
    { id: 'check-1', stage: 'check', trigger: /deploy/, confidence: 0.99, action: 'precheck', checkOutcome: 'check_fail' },
  ])
  const db = new MumpixDbAdapter([
    { name: 'db_answer', canHandle: (input) => input === 'db query', execute: () => ({ ok: true }) },
  ])
  return new RuntimeCore(
    dna,
    db,
    new ExecutionRegistry(),
    new IdempotencyRegistry(policy.idempotency),
    new TaskRegistry(),
    policy,
  )
}

test('CHECK is not optional and blocks on check_fail', async () => {
  const runtime = buildRuntime()
  const out = await runtime.handleInput('deploy this')
  assert.equal(out.source, 'dna_check')
  assert.equal(out.checkOutcome, 'check_fail')
  assert.equal(out.prevented, true)
})

test('DB operator is not skipped when computable', async () => {
  const runtime = buildRuntime()
  const out = await runtime.handleInput('db query')
  assert.equal(out.source, 'db_execute')
  assert.deepEqual(out.output, { ok: true })
  assert.equal(out.determinism.level, 'deterministic')
})

test('force_freeform rejects missing required fields', async () => {
  const runtime = buildRuntime()
  await assert.rejects(() => runtime.handleInput('unknown', { force_freeform: true }), /force_freeform requires/)
})

test('bounded fallback is evaluated before unrestricted fallback', async () => {
  const runtime = buildRuntime()
  const bounded = await runtime.handleInput('unknown path')
  assert.equal(bounded.source, 'turbo_assist')
  assert.equal(bounded.determinism.level, 'bounded')

  const longInput = 'x'.repeat(5000)
  const unrestricted = await runtime.handleInput(longInput)
  assert.equal(unrestricted.source, 'llm_freeform')
  assert.equal(unrestricted.determinism.level, 'unrestricted')
})

test('force_freeform is tracked as separate source', async () => {
  const runtime = buildRuntime()
  const out = await runtime.handleInput('totally unknown', {
    force_freeform: true,
    forceFreeformReason: 'operator override',
    callerSource: 'ops_console',
    approvedAt: Date.now(),
  })
  assert.equal(out.source, 'llm_force_freeform')
})

test('prevented action short-circuits subsequent execution in scope', async () => {
  const runtime = buildRuntime()
  const first = await runtime.handleInput('deploy this')
  const second = await runtime.handleInput('deploy this')
  assert.equal(first.prevented, true)
  assert.equal(second.shortCircuited, true)
  assert.equal(second.prevented, true)
})

test('policy denial blocks execution', async () => {
  const runtime = buildRuntime({
    validate: () => ({ allowed: false, reason: 'policy_denied' }),
  })
  await assert.rejects(() => runtime.handleInput('db query'), /Policy violation/)
})

test('timeout budget aborts execution', async () => {
  const runtime = buildRuntime({ maxExecutionMs: -1 })
  await assert.rejects(() => runtime.handleInput('db query'), /timed out/i)
})
