export interface RoutingPolicy {
  allowForceFreeform(domain: string, callerSource: string): boolean
  requireStrictDeterminism(domain: string): boolean
}

export interface IdempotencyPolicy {
  semanticTimestampFields: string[]
  transientFields: string[]
  getScopeForAction(action: string): 'session' | 'task' | 'global'
  isReExecutable(action: string): boolean
  allowSpecialNumeric(fieldPath: string): boolean
}

export interface DreamPolicy {
  maxDeferralCount: number
  maintenanceWindowFallback: boolean
  starvingGeneAlertThreshold: number
  allowedMutationOperations: Array<'promotion' | 'decay' | 'prune'>
  allowedGeneDomains?: string[]
  maintenanceWindowActive(): boolean
}

export interface RuntimePolicy {
  routing: RoutingPolicy
  idempotency: IdempotencyPolicy
  dream: DreamPolicy
}

export const defaultRuntimePolicy: RuntimePolicy = {
  routing: {
    allowForceFreeform: () => false,
    requireStrictDeterminism: () => true,
  },
  idempotency: {
    semanticTimestampFields: ['scheduled_for', 'effective_date'],
    transientFields: ['request_id', 'trace_id', 'correlation_id', 'client_timestamp'],
    getScopeForAction: () => 'session',
    isReExecutable: () => false,
    allowSpecialNumeric: () => false,
  },
  dream: {
    maxDeferralCount: 8,
    maintenanceWindowFallback: false,
    starvingGeneAlertThreshold: 5,
    allowedMutationOperations: ['promotion', 'decay', 'prune'],
    maintenanceWindowActive: () => false,
  },
}
