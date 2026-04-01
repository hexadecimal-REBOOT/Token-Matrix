import { RuntimePolicy } from '../policy/RuntimePolicy'
import { nextId } from '../shared/ids'
import { TaskRegistry } from '../tasks/TaskRegistry'

type MutationOperation = 'promotion' | 'decay' | 'prune'

export type DeferralRecord = {
  geneId: string
  deferredCount: number
  firstDeferredAt: number
  lastDeferredAt: number
  starving: boolean
}

export type DreamTaskState = {
  id: string
  genesDeferred: string[]
  genesStarving: string[]
  filesTouched: string[]
  alerts: string[]
}

export class DreamEngine {
  private readonly deferred = new Map<string, DeferralRecord>()
  private readonly operatorAlerts: string[] = []

  constructor(
    private readonly policy: RuntimePolicy,
    private readonly taskRegistry: TaskRegistry,
    private readonly getActiveGeneSet: () => Set<string>,
  ) {}

  async start(): Promise<string> {
    return this.taskRegistry.register({ type: 'dream', title: 'Dream cycle', status: 'running', startTime: Date.now() })
  }

  async runMutationBatch(taskId: string, operations: Array<{ geneId: string; domain?: string; operation: MutationOperation }>): Promise<DreamTaskState> {
    const touched: string[] = []
    for (const op of operations) {
      const result = this.tryMutation(op)
      if (result.written) touched.push(op.geneId)
    }

    this.taskRegistry.complete(taskId)
    return {
      id: taskId,
      genesDeferred: this.getDeferred().map((d) => d.geneId),
      genesStarving: this.getStarving().map((d) => d.geneId),
      filesTouched: touched,
      alerts: [...this.operatorAlerts],
    }
  }

  failTask(taskId: string, error: string): void {
    this.taskRegistry.fail(taskId, error)
  }

  tryMutation(input: { geneId: string; domain?: string; operation: MutationOperation }): { written: boolean; starving: boolean } {
    if (!this.policy.dream.allowedMutationOperations.includes(input.operation)) {
      return { written: false, starving: false }
    }

    if (this.policy.dream.allowedGeneDomains && input.domain && !this.policy.dream.allowedGeneDomains.includes(input.domain)) {
      return { written: false, starving: false }
    }

    const active = this.getActiveGeneSet()
    if (active.has(input.geneId)) {
      const record = this.deferGene(input.geneId)
      const shouldOverride =
        this.policy.dream.maintenanceWindowFallback &&
        this.policy.dream.maintenanceWindowActive() &&
        record.deferredCount >= this.policy.dream.maxDeferralCount
      if (!shouldOverride) return { written: false, starving: record.starving }
    }

    this.deferred.delete(input.geneId)
    return { written: true, starving: false }
  }

  getDeferred(): DeferralRecord[] {
    return [...this.deferred.values()]
  }

  getStarving(): DeferralRecord[] {
    return this.getDeferred().filter((d) => d.starving)
  }

  getAlerts(): string[] {
    return [...this.operatorAlerts]
  }

  private deferGene(geneId: string): DeferralRecord {
    const now = Date.now()
    const prior = this.deferred.get(geneId)
    const next: DeferralRecord = prior
      ? {
          ...prior,
          deferredCount: prior.deferredCount + 1,
          lastDeferredAt: now,
          starving: prior.deferredCount + 1 >= this.policy.dream.starvingGeneAlertThreshold,
        }
      : {
          geneId,
          deferredCount: 1,
          firstDeferredAt: now,
          lastDeferredAt: now,
          starving: false,
        }

    if (next.starving) {
      this.operatorAlerts.push(`Gene ${geneId} is starving after ${next.deferredCount} deferrals`)
    }

    this.deferred.set(geneId, next)
    return next
  }
}
