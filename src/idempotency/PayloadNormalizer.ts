import { IdempotencyPolicy } from '../policy/RuntimePolicy'

function isNumericSpecial(v: number): boolean {
  return Number.isNaN(v) || !Number.isFinite(v)
}

function normalizeNumber(v: number): number {
  if (Object.is(v, -0)) return 0
  return Number(v)
}

export class PayloadNormalizer {
  constructor(private readonly policy: IdempotencyPolicy) {}

  normalize(payload: Record<string, unknown> | undefined): string {
    const normalized = this.walk(payload ?? {}, '')
    return JSON.stringify(normalized)
  }

  private walk(value: unknown, path: string): unknown {
    if (value === undefined) return undefined
    if (value === null) return null

    if (typeof value === 'number') {
      if (isNumericSpecial(value) && !this.policy.allowSpecialNumeric(path)) {
        throw new Error(`Special numeric value is not allowed at ${path || '<root>'}`)
      }
      return normalizeNumber(value)
    }

    if (typeof value === 'string' || typeof value === 'boolean') {
      return value
    }

    if (Array.isArray(value)) {
      return value.map((item, index) => this.walk(item, `${path}[${index}]`))
    }

    if (typeof value === 'object') {
      const out: Record<string, unknown> = {}
      const obj = value as Record<string, unknown>
      for (const key of Object.keys(obj).sort()) {
        if (this.policy.transientFields.includes(key)) continue
        if (this.looksLikeTimestampField(key) && !this.policy.semanticTimestampFields.includes(key)) continue

        const next = this.walk(obj[key], path ? `${path}.${key}` : key)
        if (next !== undefined) out[key] = next
      }
      return out
    }

    throw new Error(`Unsupported payload value at ${path || '<root>'}`)
  }

  private looksLikeTimestampField(field: string): boolean {
    return /time|date|timestamp|at$/i.test(field)
  }
}
