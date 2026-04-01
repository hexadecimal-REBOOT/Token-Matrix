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
  canRetryFailed(action: string, scope: 'session' | 'task' | 'global'): boolean
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
    allowForceFreeform: (domain: string, callerSource: string) => Boolean(domain && callerSource && domain !== 'payments'),
    requireStrictDeterminism: () => true,
  },
  idempotency: {
    semanticTimestampFields: ['scheduled_for', 'effective_date'],
    transientFields: ['request_id', 'trace_id', 'correlation_id', 'client_timestamp'],
    getScopeForAction: (action: string) => (action.startsWith('fleet_') ? 'global' : 'session'),
    isReExecutable: (action: string) => action.endsWith('_read'),
    allowSpecialNumeric: () => false,
    canRetryFailed: (_action, _scope) => true,
  },
  dream: {
    maxDeferralCount: 8,
    maintenanceWindowFallback: false,
    starvingGeneAlertThreshold: 5,
    allowedMutationOperations: ['promotion', 'decay', 'prune'],
    allowedGeneDomains: undefined,
    maintenanceWindowActive: () => false,
  },
}
