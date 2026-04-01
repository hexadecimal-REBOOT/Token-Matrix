import test from 'node:test'
import assert from 'node:assert/strict'

import { PayloadNormalizer } from '../src/idempotency/PayloadNormalizer.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'

const normalizer = new PayloadNormalizer(defaultRuntimePolicy.idempotency)

test('recursive key sorting', () => {
  const json = normalizer.normalize({ b: 1, a: { z: 1, y: 2 } })
  assert.equal(json, '{"a":{"y":2,"z":1},"b":1}')
})

test('array order preservation', () => {
  const json = normalizer.normalize({ a: [3, 2, 1] })
  assert.equal(json, '{"a":[3,2,1]}')
})

test('dropping undefined and retaining null', () => {
  const json = normalizer.normalize({ a: undefined, b: null })
  assert.equal(json, '{"b":null}')
})

test('1 and 1.0 normalize the same and -0 becomes 0', () => {
  const one = normalizer.normalize({ v: 1 })
  const oneFloat = normalizer.normalize({ v: 1.0 })
  const negZero = normalizer.normalize({ v: -0 })
  assert.equal(one, oneFloat)
  assert.equal(negZero, '{"v":0}')
})

test('reject NaN and Infinity', () => {
  assert.throws(() => normalizer.normalize({ n: Number.NaN }), /Special numeric/)
  assert.throws(() => normalizer.normalize({ n: Number.POSITIVE_INFINITY }), /Special numeric/)
})

test('timestamp exclusion default + semantic inclusion + transient exclusion', () => {
  const json = normalizer.normalize({
    createdAt: 10,
    scheduled_for: '2026-01-01',
    trace_id: 'drop-me',
    nested: { eventTime: 12, ok: true },
  })

  assert.equal(json, '{"nested":{"ok":true},"scheduled_for":"2026-01-01"}')
})
