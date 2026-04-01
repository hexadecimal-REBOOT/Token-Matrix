export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'killed'

export type TaskStateBase = {
  id: string
  type: string
  title: string
  status: TaskStatus
  createdAt: number
  startTime?: number
  endTime?: number
  notified: boolean
  error?: string
  executionId?: string
}

export type DecisionSource =
  | 'dna_block'
  | 'dna_check'
  | 'dna_do'
  | 'db_execute'
  | 'turbo_assist'
  | 'llm_freeform'
  | 'llm_force_freeform'
  | 'replay'

export type DeterminismLevel = 'deterministic' | 'bounded' | 'unrestricted'
export type CheckOutcome = 'check_pass' | 'check_fail' | 'prerequisite_required'

export type RuntimeOptions = {
  force_freeform?: boolean
  forceFreeformReason?: string
  callerSource?: string
  approvedAt?: number
  sessionId?: string
  taskId?: string
  scope?: 'session' | 'task' | 'global'
  domain?: string
  timeoutMs?: number
}

export type CostBreakdown = {
  deterministic: number
  bounded: number
  unrestricted: number
  total: number
}

export type RuntimeResult = {
  executionId: string
  source: DecisionSource
  action: string
  output?: unknown
  prevented?: boolean
  checkOutcome?: CheckOutcome
  shortCircuited?: boolean
  fallbackReason?: string
  determinism: {
    level: DeterminismLevel
    reason?: string
  }
  cost: CostBreakdown
}

export type ReplayContext = {
  operatorVersion: string
  schemaVersion: string
  runtimeVersion: string
}

export type ExecutionRecord = {
  id: string
  executionId: string
  sessionId: string
  taskId?: string
  timestamp: number
  durationMs?: number
  idempotencyKey: string
  determinism: {
    level: DeterminismLevel
    reason?: string
  }
  replayContext?: ReplayContext
  cost: CostBreakdown
  input: {
    raw: string
    parsedIntent?: string
    domain?: string
  }
  routing: {
    source: DecisionSource
    matchedGeneId?: string
    checkOutcome?: CheckOutcome
    operator?: string
    fallbackReason?: string
    forceFreeform?: {
      reason: string
      callerSource: string
      approvedAt: number
    }
  }
  action: {
    name: string
    args?: Record<string, unknown>
    reExecutable?: boolean
  }
  result: {
    success: boolean
    output?: unknown
    error?: string
    prevented?: boolean
    shortCircuited?: boolean
  }
  outcome: {
    reward?: number
    damage?: number
    tags?: string[]
    trigger?: string
    walPatternClass?: string
  }
}

export type ReplayCandidate = {
  recordId: string
  actionSequence: string[]
  operatorSequence: string[]
  signature: string
  walPatternClass?: string
  replayContext?: ReplayContext
}

export type IdempotencyStatus = 'not_seen' | 'claimed' | 'in_flight' | 'completed' | 'failed'

export type IdempotencyRecord = {
  key: string
  action: string
  scope: 'session' | 'task' | 'global'
  status: IdempotencyStatus
  startedAt?: number
  completedAt?: number
  result?: unknown
  error?: string
  prevented?: boolean
  preventedReason?: string
  executionId?: string
}

export type OperatorGapEvent = {
  intent: string
  domain: string
  payloadShape: string[]
  reason: 'no_operator_match'
}

export class RuntimeInvariantError extends Error {}
export class TimeoutError extends Error {}
export class StepLimitError extends Error {}
