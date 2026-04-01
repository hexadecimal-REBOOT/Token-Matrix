import { nextId } from '../shared/ids'
import { CheckOutcome, DecisionSource, ExecutionRecord, ReplayCandidate, ReplayContext } from '../shared/types'

export type CoverageStats = {
  total: number
  bySource: Record<string, number>
  checkOutcomes: Record<string, number>
  boundedFallbackRate: number
  unrestrictedFallbackRate: number
  forceFreeformRate: number
}

export interface IExecutionRegistry {
  append(record: Omit<ExecutionRecord, 'id' | 'timestamp'>): string
  get(id: string): ExecutionRecord | undefined
  listBySource(source: DecisionSource): ExecutionRecord[]
  listByCheckOutcome(outcome: CheckOutcome): ExecutionRecord[]
  getCoverageStats(): CoverageStats
  getReplayCandidates(action: string, opts?: { embeddingQuery?: string; expectedContext?: ReplayContext }): ReplayCandidate[]
  explain(id: string): string
  replayTimeline(id: string): Array<{ step: number; text: string }>
}

export class ExecutionRegistry implements IExecutionRegistry {
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
    const freeform = bySource.llm_freeform ?? 0
    const forceFreeform = bySource.llm_force_freeform ?? 0

    return {
      total,
      bySource,
      checkOutcomes,
      boundedFallbackRate: total ? bounded / total : 0,
      unrestrictedFallbackRate: total ? (freeform + forceFreeform) / total : 0,
      forceFreeformRate: total ? forceFreeform / total : 0,
    }
  }

  getReplayCandidates(
    action: string,
    opts?: { embeddingQuery?: string; expectedContext?: ReplayContext },
  ): ReplayCandidate[] {
    if (opts?.embeddingQuery) {
      throw new Error('Embedding lookup is forbidden in deterministic replay path')
    }

    const result: ReplayCandidate[] = []
    for (const record of this.records) {
      if (record.action.name !== action) continue
      if (opts?.expectedContext && !this.contextMatches(record.replayContext, opts.expectedContext)) {
        continue
      }

      result.push({
        recordId: record.id,
        actionSequence: [record.action.name],
        operatorSequence: record.routing.operator ? [record.routing.operator] : [],
        signature: `${record.input.domain ?? 'unknown'}:${record.outcome.trigger ?? 'none'}:${record.result.success ? 'success' : 'fail'}`,
        walPatternClass: record.outcome.walPatternClass,
        replayContext: record.replayContext,
      })
    }

    return result
  }



  replayTimeline(id: string): Array<{ step: number; text: string }> {
    const r = this.get(id)
    if (!r) return []
    return [
      { step: 1, text: `Input received in domain ${r.input.domain ?? 'unknown'}` },
      { step: 2, text: `Routing source selected: ${r.routing.source}` },
      { step: 3, text: `Action executed: ${r.action.name}` },
      { step: 4, text: `Result: ${r.result.success ? 'success' : 'failure'}` },
    ]
  }

  explain(id: string): string {
    const record = this.get(id)
    if (!record) return `Execution ${id} not found`
    if (record.determinism.level === 'deterministic') {
      return `Used deterministic ${record.routing.source}${record.routing.operator ? ` operator ${record.routing.operator}` : ''}`
    }
    return `Fell back to ${record.routing.source} because ${record.routing.fallbackReason ?? 'fallback policy'}`
  }

  private contextMatches(recordCtx: ReplayContext | undefined, expected: ReplayContext): boolean {
    if (!recordCtx) return false
    return (
      recordCtx.operatorVersion === expected.operatorVersion &&
      recordCtx.schemaVersion === expected.schemaVersion &&
      recordCtx.runtimeVersion === expected.runtimeVersion
    )
  }
}
