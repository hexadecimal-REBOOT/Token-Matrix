import { RuntimePolicy } from '../policy/RuntimePolicy'
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
  snapshotVersion: number
  finalVersion: number
}

export type DreamImpact = {
  geneId: string
  beforeDeterministicRate: number
  afterDeterministicRate: number
  executionsAffected: number
}

export type DreamTransaction = {
  readSnapshotVersion: number
  writeBatch: Array<{ geneId: string; domain?: string; operation: MutationOperation }>
}

export class DreamEngine {
  private readonly deferred = new Map<string, DeferralRecord>()
  private readonly operatorAlerts: string[] = []
  private stateVersion = 0
  private readonly impacts: DreamImpact[] = []

  constructor(
    private readonly policy: RuntimePolicy,
    private readonly taskRegistry: TaskRegistry,
    private readonly getActiveGeneSet: () => Set<string>,
  ) {}

  async start(executionId?: string): Promise<string> {
    return this.taskRegistry.register({
      type: 'dream',
      title: 'Dream cycle',
      status: 'running',
      startTime: Date.now(),
      executionId,
    })
  }

  beginTransaction(writeBatch: DreamTransaction['writeBatch']): DreamTransaction {
    return {
      readSnapshotVersion: this.stateVersion,
      writeBatch,
    }
  }

  async commitTransaction(taskId: string, tx: DreamTransaction): Promise<DreamTaskState> {
    if (tx.readSnapshotVersion !== this.stateVersion) {
      throw new Error('Dream write isolation violation: state changed since snapshot')
    }

    const touched: string[] = []
    for (const op of tx.writeBatch) {
      const result = this.tryMutation(op)
      if (result.written) touched.push(op.geneId)
    }

    this.stateVersion += 1
    this.taskRegistry.complete(taskId)
    return {
      id: taskId,
      genesDeferred: this.getDeferred().map((d) => d.geneId),
      genesStarving: this.getStarving().map((d) => d.geneId),
      filesTouched: touched,
      alerts: [...this.operatorAlerts],
      snapshotVersion: tx.readSnapshotVersion,
      finalVersion: this.stateVersion,
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

  getStateVersion(): number {
    return this.stateVersion
  }

  recordImpact(impact: DreamImpact): void {
    this.impacts.push(impact)
  }

  getImpacts(): DreamImpact[] {
    return [...this.impacts]
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
