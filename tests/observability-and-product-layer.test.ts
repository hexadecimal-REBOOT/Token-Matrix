import test from 'node:test'
import assert from 'node:assert/strict'

import { MumpixDbAdapter } from '../src/db/MumpixDbAdapter.ts'
import { DNARuntime } from '../src/dna/DNARuntime.ts'
import { ExecutionRegistry } from '../src/execution/ExecutionRegistry.ts'
import { IdempotencyRegistry } from '../src/idempotency/IdempotencyRegistry.ts'
import { RuntimeMetrics } from '../src/metrics/RuntimeMetrics.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'
import { RuntimeCore } from '../src/runtime/RuntimeCore.ts'
import { OperatorGapRegistry } from '../src/runtime/OperatorGapRegistry.ts'
import { TaskRegistry } from '../src/tasks/TaskRegistry.ts'

test('runtime metrics aggregate rates, costs, and fallback reasons', async () => {
  const execution = new ExecutionRegistry()
  const metrics = new RuntimeMetrics()
  const runtime = new RuntimeCore(
    new DNARuntime(),
    new MumpixDbAdapter(),
    execution,
    new IdempotencyRegistry(defaultRuntimePolicy.idempotency),
    new TaskRegistry(),
    defaultRuntimePolicy,
    metrics,
    new OperatorGapRegistry(),
  )

  await runtime.handleInput('unknown-a')
  await runtime.handleInput('x'.repeat(5001))

  const snapshot = runtime.getMetrics()
  assert.equal(snapshot.totalExecutions, 2)
  assert.ok(snapshot.boundedRate > 0)
  assert.ok(snapshot.unrestrictedRate > 0)
  assert.ok(snapshot.costs.total > 0)
})

test('operator gaps emitted when no deterministic operator matches', async () => {
  const gaps = new OperatorGapRegistry()
  const runtime = new RuntimeCore(
    new DNARuntime(),
    new MumpixDbAdapter(),
    new ExecutionRegistry(),
    new IdempotencyRegistry(defaultRuntimePolicy.idempotency),
    new TaskRegistry(),
    defaultRuntimePolicy,
    new RuntimeMetrics(),
    gaps,
  )

  await runtime.handleInput('u'.repeat(5001))
  assert.equal(gaps.list().at(-1)?.reason, 'no_operator_match')
})

test('strict mode rejects unrestricted fallback', async () => {
  const runtime = new RuntimeCore(
    new DNARuntime(),
    new MumpixDbAdapter(),
    new ExecutionRegistry(),
    new IdempotencyRegistry(defaultRuntimePolicy.idempotency),
    new TaskRegistry(),
    { ...defaultRuntimePolicy, strictMode: true },
    new RuntimeMetrics(),
    new OperatorGapRegistry(),
  )

  await assert.rejects(() => runtime.handleInput('x'.repeat(5001)), /Strict mode/)
})

test('execution explain returns human-readable audit', () => {
  const ex = new ExecutionRegistry()
  const id = ex.append({
    executionId: 'e1',
    sessionId: 's1',
    idempotencyKey: 'k',
    determinism: { level: 'deterministic' },
    cost: { deterministic: 0, bounded: 0, unrestricted: 0, total: 0 },
    input: { raw: 'x', domain: 'ops' },
    routing: { source: 'db_execute', operator: 'timeline.order' },
    action: { name: 'compute' },
    result: { success: true },
    outcome: {},
  })

  assert.match(ex.explain(id), /deterministic/)
})
