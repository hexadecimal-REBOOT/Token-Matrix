import { RuntimePolicy } from '../policy/RuntimePolicy'
import { nextId } from '../shared/ids'

export type DeferralRecord = {
  geneId: string
  deferredCount: number
  firstDeferredAt: number
  lastDeferredAt: number
  starving: boolean
}

export class DreamEngine {
  private readonly deferred = new Map<string, DeferralRecord>()

  constructor(private readonly policy: RuntimePolicy, private readonly getActiveGeneSet: () => Set<string>) {}

  async start(): Promise<string> {
    return nextId('dream')
  }

  async sweep(): Promise<{ deferred: number }> {
    return { deferred: this.deferred.size }
  }

  tryMutation(geneId: string): { written: boolean; starving: boolean } {
    const active = this.getActiveGeneSet()
    if (active.has(geneId)) {
      const now = Date.now()
      const prior = this.deferred.get(geneId)
      const record: DeferralRecord = prior
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

      const shouldOverride =
        this.policy.dream.maintenanceWindowFallback &&
        this.policy.dream.maintenanceWindowActive() &&
        record.deferredCount >= this.policy.dream.maxDeferralCount

      if (!shouldOverride) {
        this.deferred.set(geneId, record)
        return { written: false, starving: record.starving }
      }
    }

    this.deferred.delete(geneId)
    return { written: true, starving: false }
  }

  getDeferred(): DeferralRecord[] {
    return [...this.deferred.values()]
  }

  getStarving(): DeferralRecord[] {
    return this.getDeferred().filter((d) => d.starving)
  }
}
