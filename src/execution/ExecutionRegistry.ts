import { nextId } from '../shared/ids'
import { CheckOutcome, DecisionSource, ExecutionRecord } from '../shared/types'

export type CoverageStats = {
  total: number
  bySource: Record<string, number>
  checkOutcomes: Record<string, number>
  boundedFallbackRate: number
  unrestrictedFallbackRate: number
}

export class ExecutionRegistry {
  private readonly records: ExecutionRecord[] = []

  append(record: Omit<ExecutionRecord, 'id' | 'timestamp'>): string {
    const id = nextId('exec')
    this.records.push({ ...record, id, timestamp: Date.now() })
    return id
  }

  get(id: string): ExecutionRecord | undefined {
    return this.records.find((r) => r.id === id)
  }

  listBySession(sessionId: string): ExecutionRecord[] {
    return this.records.filter((r) => r.sessionId === sessionId)
  }

  listByTask(taskId: string): ExecutionRecord[] {
    return this.records.filter((r) => r.taskId === taskId)
  }

  listBySource(source: DecisionSource): ExecutionRecord[] {
    return this.records.filter((r) => r.routing.source === source)
  }

  listByCheckOutcome(outcome: CheckOutcome): ExecutionRecord[] {
    return this.records.filter((r) => r.routing.checkOutcome === outcome)
  }

  getCoverageStats(): CoverageStats {
    const total = this.records.length
    const bySource: Record<string, number> = {}
    const checkOutcomes: Record<string, number> = {}

    for (const record of this.records) {
      bySource[record.routing.source] = (bySource[record.routing.source] ?? 0) + 1
      if (record.routing.checkOutcome) {
        checkOutcomes[record.routing.checkOutcome] = (checkOutcomes[record.routing.checkOutcome] ?? 0) + 1
      }
    }

    const bounded = bySource.turbo_assist ?? 0
    const unrestricted = (bySource.llm_freeform ?? 0) + (bySource.llm_force_freeform ?? 0)

    return {
      total,
      bySource,
      checkOutcomes,
      boundedFallbackRate: total ? bounded / total : 0,
      unrestrictedFallbackRate: total ? unrestricted / total : 0,
    }
  }
}
