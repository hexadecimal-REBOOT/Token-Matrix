import { DecisionSource } from '../shared/types'

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
  maxExecutionMs: number
  maxSteps: number
  maxFallbackDepth: number
  runtimeVersion: string
  operatorVersion: string
  schemaVersion: string
  validate(action: string, domain: string, source: DecisionSource): { allowed: boolean; reason?: string }
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
  maxExecutionMs: 5_000,
  maxSteps: 8,
  maxFallbackDepth: 2,
  runtimeVersion: '0.4.0',
  operatorVersion: '1',
  schemaVersion: '1',
  validate: (_action, _domain, _source) => ({ allowed: true }),
}
