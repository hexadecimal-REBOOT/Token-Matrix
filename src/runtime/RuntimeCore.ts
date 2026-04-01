import { MumpixDbAdapter } from '../db/MumpixDbAdapter'
import { DNARuntime } from '../dna/DNARuntime'
import { ExecutionRegistry, IExecutionRegistry } from '../execution/ExecutionRegistry'
import { IIdempotencyRegistry, IdempotencyRegistry } from '../idempotency/IdempotencyRegistry'
import { estimateCost, RuntimeMetrics } from '../metrics/RuntimeMetrics'
import { RuntimePolicy } from '../policy/RuntimePolicy'
import { ITaskRegistry, TaskRegistry } from '../tasks/TaskRegistry'
import { StepLimitError, TimeoutError, RuntimeInvariantError, RuntimeOptions, RuntimeResult } from '../shared/types'
import { nextId } from '../shared/ids'
import { ContextAssembler } from './ContextAssembler'
import { DecisionRouter } from './DecisionRouter'
import { IOperatorGapRegistry, OperatorGapRegistry } from './OperatorGapRegistry'
import { SessionRuntime } from './SessionRuntime'
import { ICoordinationHook, InMemoryCoordinationHook } from './hooks/CoordinationHook'

export class RuntimeCore {
  private readonly router: DecisionRouter
  private readonly sessionRuntime: SessionRuntime
  private readonly contextAssembler = new ContextAssembler()

  constructor(
    private readonly dna: DNARuntime,
    private readonly db: MumpixDbAdapter,
    private readonly execution: IExecutionRegistry,
    private readonly idempotency: IIdempotencyRegistry,
    private readonly tasks: ITaskRegistry,
    private readonly policy: RuntimePolicy,
    private readonly metrics: RuntimeMetrics = new RuntimeMetrics(),
    private readonly operatorGaps: IOperatorGapRegistry = new OperatorGapRegistry(),
    private readonly coordination: ICoordinationHook = new InMemoryCoordinationHook(),
  ) {
    this.router = new DecisionRouter(dna, db)
    this.sessionRuntime = new SessionRuntime(dna)
  }

  async handleInput(input: string, opts: RuntimeOptions = {}): Promise<RuntimeResult> {
    const executionId = nextId('run')
    const startedAt = Date.now()
    const timeoutMs = opts.timeoutMs ?? this.policy.maxExecutionMs
    let steps = 0
    let fallbackDepth = 0
    const trace = {
      attempted: { deterministic: false, bounded: false, unrestricted: false },
    }

    const step = () => {
      steps += 1
      if (steps > this.policy.maxSteps) throw new StepLimitError(`Step limit exceeded: ${steps}`)
      if (Date.now() - startedAt > timeoutMs) throw new TimeoutError(`Execution timed out after ${Date.now() - startedAt}ms`)
    }

    const sessionId = opts.sessionId ?? 'default'
    const domain = opts.domain ?? 'general'
    const session = this.sessionRuntime.getOrCreate(sessionId)
    this.sessionRuntime.pushHistory(sessionId, input)

    const forceDeclared = Boolean(opts.force_freeform)
    if (forceDeclared) this.validateForceFreeform(opts, domain)

    step()
    trace.attempted.deterministic = true
    const deterministic = this.router.nextDeterministic(input)

    let decision = deterministic
    if (!decision && forceDeclared) {
      step()
      trace.attempted.unrestricted = true
      decision = this.router.nextUnrestrictedFallback(true)
    }

    if (!decision) {
      step()
      fallbackDepth += 1
      trace.attempted.bounded = true
      const bounded = this.router.nextBoundedFallback()
      const boundedResult = this.simulateTurboAssist(input, session.id)
      decision = boundedResult !== undefined ? bounded : undefined
      if (!boundedResult) {
        this.operatorGaps.emit({ intent: input, domain, payloadShape: Object.keys({ input, domain }), reason: 'no_operator_match' })
      }

      if (!decision) {
        step()
        fallbackDepth += 1
        trace.attempted.unrestricted = true
        if (!trace.attempted.bounded) {
          throw new RuntimeInvariantError('Invariant 1 violation: unrestricted fallback before bounded attempt')
        }
        decision = this.router.nextUnrestrictedFallback(false)
      }
    }

    if (this.policy.strictMode && (decision.source === 'llm_freeform' || decision.source === 'llm_force_freeform')) {
      throw new RuntimeInvariantError('Strict mode forbids unrestricted fallback')
    }

    if (fallbackDepth > this.policy.maxFallbackDepth) throw new StepLimitError(`Fallback depth exceeded: ${fallbackDepth}`)
    if (!trace.attempted.deterministic) throw new RuntimeInvariantError('Invariant 1 violation: deterministic tier not attempted')

    const scope = opts.scope ?? this.policy.idempotency.getScopeForAction(decision.action)
    const key = this.idempotency.computeKey({
      action: decision.action,
      payload: { input, source: decision.source, domain },
      scope,
      sessionId,
      taskId: opts.taskId,
    })

    const claim = this.idempotency.checkAndClaim({ key, action: decision.action, scope, executionId })
    if (claim.status === 'completed' && claim.record) {
      this.metrics.recordIdempotencyHit()
      return {
        executionId,
        source: decision.source,
        action: decision.action,
        output: claim.record.result,
        prevented: claim.record.prevented,
        shortCircuited: true,
        determinism: this.toDeterminism(decision.source, decision.fallbackReason),
        cost: estimateCost(decision.source),
      }
    }

    if (claim.status === 'in_flight') {
      this.metrics.recordInFlightCollision()
      throw new RuntimeInvariantError(`Action already in flight for key ${key}`)
    }

    if (claim.status === 'failed' && !this.policy.idempotency.canRetryFailed(decision.action, scope)) {
      throw new RuntimeInvariantError(`Retry denied by policy for failed action ${decision.action}`)
    }

    const validation = this.policy.validate(decision.action, domain, decision.source)
    if (!validation.allowed) {
      throw new RuntimeInvariantError(`Policy violation: ${validation.reason ?? 'policy_denied'}`)
    }

    try {
      step()
      const result = this.executeDecision(input, decision, executionId)
      const cost = estimateCost(result.source)
      this.idempotency.complete(key, result.output, {
        prevented: result.prevented,
        preventedReason: result.prevented ? `Prevented by ${decision.source}` : undefined,
      })

      const record = {
        executionId,
        sessionId,
        taskId: opts.taskId,
        durationMs: Date.now() - startedAt,
        idempotencyKey: key,
        determinism: result.determinism,
        replayContext: {
          operatorVersion: this.policy.operatorVersion,
          schemaVersion: this.policy.schemaVersion,
          runtimeVersion: this.policy.runtimeVersion,
        },
        cost,
        input: { raw: input, domain },
        routing: {
          source: result.source,
          matchedGeneId: decision.matchedGeneId,
          checkOutcome: result.checkOutcome,
          operator: decision.operator,
          fallbackReason: result.fallbackReason,
          forceFreeform: forceDeclared
            ? {
                reason: opts.forceFreeformReason!,
                callerSource: opts.callerSource!,
                approvedAt: opts.approvedAt!,
              }
            : undefined,
        },
        action: {
          name: decision.action,
          reExecutable: this.policy.idempotency.isReExecutable(decision.action),
        },
        result: {
          success: true,
          output: result.output,
          prevented: result.prevented,
          shortCircuited: result.shortCircuited,
        },
        outcome: {},
      }
      const id = this.execution.append(record)
      this.metrics.recordExecution({ ...record, id, timestamp: Date.now() })
      this.coordination.publish({ type: 'execution_completed', executionId, recordId: id })

      return { ...result, cost }
    } catch (error) {
      this.idempotency.fail(key, error instanceof Error ? error.message : 'unknown')
      throw error
    }
  }

  getSession(sessionId: string) {
    return this.sessionRuntime.get(sessionId)
  }

  getMetrics() {
    return this.metrics.snapshot()
  }

  onMetricEvent(listener: Parameters<RuntimeMetrics['subscribe']>[0]) {
    return this.metrics.subscribe(listener)
  }

  getOperatorGaps() {
    return this.operatorGaps.list()
  }

  async startBackgroundTask(type: string): Promise<string> {
    const taskId = this.tasks.register({ type, title: `${type} task`, status: 'running', startTime: Date.now() })
    this.coordination.publish({ type: 'task_started', taskId })
    return taskId
  }

  replay(action: string): Promise<{ action: string; replayed: boolean }> {
    const candidates = this.execution.getReplayCandidates(action, {
      expectedContext: {
        operatorVersion: this.policy.operatorVersion,
        schemaVersion: this.policy.schemaVersion,
        runtimeVersion: this.policy.runtimeVersion,
      },
    })
    return Promise.resolve({ action, replayed: candidates.length > 0 })
  }

  private executeDecision(
    input: string,
    decision: NonNullable<ReturnType<DecisionRouter['nextDeterministic']>> | ReturnType<DecisionRouter['nextBoundedFallback']> | ReturnType<DecisionRouter['nextUnrestrictedFallback']>,
    executionId: string,
  ): RuntimeResult {
    if (!decision) throw new RuntimeInvariantError('No routing decision selected')

    if (decision.source === 'dna_check' && decision.checkOutcome === 'prerequisite_required') {
      return {
        executionId,
        source: decision.source,
        action: decision.action,
        checkOutcome: 'prerequisite_required',
        output: { resolutionRequired: true },
        determinism: this.toDeterminism(decision.source, 'prerequisite_required'),
        cost: estimateCost(decision.source),
      }
    }

    if (decision.source === 'dna_check' && decision.checkOutcome === 'check_fail') {
      return {
        executionId,
        source: decision.source,
        action: decision.action,
        checkOutcome: 'check_fail',
        prevented: true,
        determinism: this.toDeterminism(decision.source, 'check_failed'),
        cost: estimateCost(decision.source),
      }
    }

    if (decision.source === 'db_execute' && decision.operator) {
      return {
        executionId,
        source: decision.source,
        action: decision.action,
        output: this.db.resolve(input)?.execute(input),
        fallbackReason: decision.fallbackReason,
        determinism: this.toDeterminism(decision.source),
        cost: estimateCost(decision.source),
      }
    }

    return {
      executionId,
      source: decision.source,
      action: decision.action,
      prevented: Boolean(decision.prevented),
      checkOutcome: decision.checkOutcome,
      fallbackReason: decision.fallbackReason,
      determinism: this.toDeterminism(decision.source, decision.fallbackReason),
      cost: estimateCost(decision.source),
    }
  }

  private simulateTurboAssist(input: string, sessionId: string): string | undefined {
    const assembled = this.contextAssembler.assemble(this.sessionRuntime.getOrCreate(sessionId), input)
    return assembled.length <= 4000 ? `turbo:${input}` : undefined
  }

  private validateForceFreeform(opts: RuntimeOptions, domain: string): void {
    if (!opts.forceFreeformReason || !opts.callerSource || !opts.approvedAt) {
      throw new RuntimeInvariantError('force_freeform requires forceFreeformReason, callerSource, and approvedAt')
    }
    if (!this.policy.routing.allowForceFreeform(domain, opts.callerSource)) {
      throw new RuntimeInvariantError(`force_freeform denied for domain ${domain}`)
    }
  }

  private toDeterminism(source: RuntimeResult['source'], reason?: string): RuntimeResult['determinism'] {
    if (source === 'turbo_assist') return { level: 'bounded', reason }
    if (source === 'llm_freeform' || source === 'llm_force_freeform') return { level: 'unrestricted', reason }
    return { level: 'deterministic', reason }
  }
}

export function createRuntimeCore(policy: RuntimePolicy) {
  const dna = new DNARuntime()
  const db = new MumpixDbAdapter()
  const execution = new ExecutionRegistry()
  const idempotency = new IdempotencyRegistry(policy.idempotency)
  const tasks = new TaskRegistry()
  const metrics = new RuntimeMetrics()
  const gaps = new OperatorGapRegistry()
  const coordination = new InMemoryCoordinationHook()
  return new RuntimeCore(dna, db, execution, idempotency, tasks, policy, metrics, gaps, coordination)
}
