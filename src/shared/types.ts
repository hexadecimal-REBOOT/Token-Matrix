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
}

export type RuntimeResult = {
  source: DecisionSource
  action: string
  output?: unknown
  prevented?: boolean
  checkOutcome?: CheckOutcome
  shortCircuited?: boolean
  fallbackReason?: string
}

export type ExecutionRecord = {
  id: string
  sessionId: string
  taskId?: string
  timestamp: number
  idempotencyKey: string
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
}

export type IdempotencyStatus = 'not_seen' | 'in_flight' | 'completed' | 'failed'

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
}

export class RuntimeInvariantError extends Error {}
