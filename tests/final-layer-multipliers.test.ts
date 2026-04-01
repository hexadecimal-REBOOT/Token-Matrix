import test from 'node:test'
import assert from 'node:assert/strict'

import { MumpixDbAdapter } from '../src/db/MumpixDbAdapter.ts'
import { DNARuntime } from '../src/dna/DNARuntime.ts'
import { ExecutionRegistry } from '../src/execution/ExecutionRegistry.ts'
import { IdempotencyRegistry } from '../src/idempotency/IdempotencyRegistry.ts'
import { RuntimeMetrics } from '../src/metrics/RuntimeMetrics.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'
import { RuntimeCore } from '../src/runtime/RuntimeCore.ts'
import { InMemoryCoordinationHook } from '../src/runtime/hooks/CoordinationHook.ts'
import { OperatorGapRegistry } from '../src/runtime/OperatorGapRegistry.ts'
import { OperatorSuggester } from '../src/suggestions/OperatorSuggester.ts'
import { TaskRegistry } from '../src/tasks/TaskRegistry.ts'

test('metrics stream emits execution events', async () => {
  const metrics = new RuntimeMetrics()
  const events: string[] = []
  const stop = metrics.subscribe((event) => events.push(event.type))

  const runtime = new RuntimeCore(
    new DNARuntime(),
    new MumpixDbAdapter(),
    new ExecutionRegistry(),
    new IdempotencyRegistry(defaultRuntimePolicy.idempotency),
    new TaskRegistry(),
    defaultRuntimePolicy,
    metrics,
    new OperatorGapRegistry(),
    new InMemoryCoordinationHook(),
  )

  await runtime.handleInput('x'.repeat(5001))
  stop()
  assert.ok(events.includes('execution_complete'))
})

test('operator suggester proposes deterministic operator', () => {
  const s = new OperatorSuggester()
  const out = s.suggestOperator({
    intent: 'Compare timeline windows',
    domain: 'temporal',
    payloadShape: ['start', 'end'],
    examples: ['compare Jan vs Feb'],
  })
  assert.match(out.suggestedName, /^temporal\./)
})

test('replay timeline returns step-by-step playback', () => {
  const ex = new ExecutionRegistry()
  const id = ex.append({
    executionId: 'e2',
    sessionId: 's2',
    idempotencyKey: 'k2',
    determinism: { level: 'bounded' },
    cost: { deterministic: 0, bounded: 0.001, unrestricted: 0, total: 0.001 },
    input: { raw: 'hello', domain: 'demo' },
    routing: { source: 'turbo_assist', fallbackReason: 'no_operator_match' },
    action: { name: 'assist' },
    result: { success: true },
    outcome: {},
  })

  const timeline = ex.replayTimeline(id)
  assert.equal(timeline.length, 4)
})

test('coordination hook captures distributed-ready events', async () => {
  const hook = new InMemoryCoordinationHook()
  const runtime = new RuntimeCore(
    new DNARuntime(),
    new MumpixDbAdapter(),
    new ExecutionRegistry(),
    new IdempotencyRegistry(defaultRuntimePolicy.idempotency),
    new TaskRegistry(),
    defaultRuntimePolicy,
    new RuntimeMetrics(),
    new OperatorGapRegistry(),
    hook,
  )

  await runtime.startBackgroundTask('dream')
  await runtime.handleInput('x'.repeat(5001))

  const kinds = hook.list().map((e) => e.type)
  assert.ok(kinds.includes('task_started'))
  assert.ok(kinds.includes('execution_completed'))
})
