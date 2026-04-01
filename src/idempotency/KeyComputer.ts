import { createHash } from 'node:crypto'
import { IdempotencyPolicy } from '../policy/RuntimePolicy'
import { PayloadNormalizer } from './PayloadNormalizer'

export class KeyComputer {
  private readonly normalizer: PayloadNormalizer

  constructor(policy: IdempotencyPolicy) {
    this.normalizer = new PayloadNormalizer(policy)
  }

  compute(input: { action: string; scope: string; payload?: Record<string, unknown> }): string {
    const canonicalPayload = this.normalizer.normalize(input.payload)
    const digest = createHash('sha256').update(Buffer.from(canonicalPayload, 'utf8')).digest('hex')
    return `${input.action}:${input.scope}:${digest}`
  }
}
