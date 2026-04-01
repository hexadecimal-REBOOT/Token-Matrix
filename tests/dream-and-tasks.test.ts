import test from 'node:test'
import assert from 'node:assert/strict'

import { DreamEngine } from '../src/dream/DreamEngine.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'
import { TaskRegistry } from '../src/tasks/TaskRegistry.ts'

test('dream defers active gene and marks starvation', async () => {
  const tasks = new TaskRegistry()
  const policy = {
    ...defaultRuntimePolicy,
    dream: {
      ...defaultRuntimePolicy.dream,
      starvingGeneAlertThreshold: 2,
    },
  }
  const dream = new DreamEngine(policy, tasks, () => new Set(['g1']))
  const taskId = await dream.start()
  dream.tryMutation({ geneId: 'g1', operation: 'promotion' })
  const second = dream.tryMutation({ geneId: 'g1', operation: 'promotion' })
  assert.equal(second.written, false)
  assert.equal(second.starving, true)
  const state = await dream.runMutationBatch(taskId, [{ geneId: 'g2', operation: 'promotion' }])
  assert.ok(state.genesDeferred.includes('g1'))
  assert.ok(state.genesStarving.includes('g1'))
})

test('task registry detects unfinished tasks', () => {
  const tasks = new TaskRegistry()
  tasks.register({ type: 'replay', title: 'Replay', status: 'running', notified: false })
  assert.throws(() => tasks.assertNoDanglingTasks(), /Spec violation/)
})
