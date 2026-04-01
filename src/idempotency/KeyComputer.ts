import { createHash } from 'node:crypto'

export function computeIdempotencyKey(input: { action: string; scope: string; canonicalPayload: string }): string {
  const digest = createHash('sha256').update(Buffer.from(input.canonicalPayload, 'utf8')).digest('hex')
  return `${input.action}:${input.scope}:${digest}`
}
