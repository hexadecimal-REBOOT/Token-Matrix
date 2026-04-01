import { MumpixDbAdapter } from '../db/MumpixDbAdapter'
import { DNARuntime } from '../dna/DNARuntime'
import { ExecutionRegistry } from '../execution/ExecutionRegistry'
import { IdempotencyRegistry } from '../idempotency/IdempotencyRegistry'
import { RuntimePolicy } from '../policy/RuntimePolicy'
import { TaskRegistry } from '../tasks/TaskRegistry'
import { RuntimeInvariantError, RuntimeOptions, RuntimeResult } from '../shared/types'
import { ContextAssembler } from './ContextAssembler'
import { DecisionRouter } from './DecisionRouter'
import { SessionRuntime } from './SessionRuntime'

export class RuntimeCore {
  private readonly router: DecisionRouter
  private readonly sessionRuntime: SessionRuntime
  private readonly contextAssembler = new ContextAssembler()

  constructor(
    private readonly dna: DNARuntime,
    private readonly db: MumpixDbAdapter,
    private readonly execution: ExecutionRegistry,
    private readonly idempotency: IdempotencyRegistry,
    private readonly tasks: TaskRegistry,
    private readonly policy: RuntimePolicy,
  ) {
    this.router = new DecisionRouter(dna, db)
    this.sessionRuntime = new SessionRuntime(dna)
  }

  async handleInput(input: string, opts: RuntimeOptions = {}): Promise<RuntimeResult> {
    const sessionId = opts.sessionId ?? 'default'
    const domain = opts.domain ?? 'general'
    const session = this.sessionRuntime.getOrCreate(sessionId)
    this.sessionRuntime.pushHistory(sessionId, input)

    const forceDeclared = Boolean(opts.force_freeform)
    if (forceDeclared) this.validateForceFreeform(opts, domain)

    const deterministic = this.router.nextDeterministic(input)
    const decision = deterministic ?? this.pickFallbackRoute(input, session.id, forceDeclared)


    const scope = opts.scope ?? this.policy.idempotency.getScopeForAction(decision.action)
    const key = this.idempotency.computeKey({
      action: decision.action,
      payload: { input, source: decision.source, domain },
      scope,
      sessionId,
      taskId: opts.taskId,
    })

    const idempotencyState = this.idempotency.check(key)
    if (idempotencyState.status === 'completed' && idempotencyState.record) {
      return {
        source: decision.source,
        action: decision.action,
        output: idempotencyState.record.result,
        prevented: idempotencyState.record.prevented,
        shortCircuited: true,
      }
    }

    if (idempotencyState.status === 'in_flight') {
      throw new RuntimeInvariantError(`Action already in flight for key ${key}`)
    }

    if (idempotencyState.status === 'failed' && !this.policy.idempotency.canRetryFailed(decision.action, scope)) {
      throw new RuntimeInvariantError(`Retry denied by policy for failed action ${decision.action}`)
    }

    this.idempotency.start(key, decision.action, scope)

    try {
      const result = this.executeDecision(input, decision)
      this.idempotency.complete(key, result.output, {
        prevented: result.prevented,
        preventedReason: result.prevented ? `Prevented by ${decision.source}` : undefined,
      })

      this.execution.append({
        sessionId,
        taskId: opts.taskId,
        idempotencyKey: key,
        input: { raw: input, domain },
        routing: {
          source: result.source,
          matchedGeneId: decision.matchedGeneId,
          checkOutcome: result.checkOutcome,
          operator: decision.operator,
          fallbackReason: decision.fallbackReason,
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
      })

      return result
    } catch (error) {
      this.idempotency.fail(key, error instanceof Error ? error.message : 'unknown')
      throw error
    }
  }

  getSession(sessionId: string) {
    return this.sessionRuntime.get(sessionId)
  }

  async startBackgroundTask(type: string): Promise<string> {
    return this.tasks.register({ type, title: `${type} task`, status: 'running', startTime: Date.now() })
  }

  replay(action: string): Promise<{ action: string; replayed: boolean }> {
    const candidates = this.execution.getReplayCandidates(action)
    return Promise.resolve({ action, replayed: candidates.length > 0 })
  }

  private executeDecision(input: string, decision: NonNullable<ReturnType<DecisionRouter['nextDeterministic']>> | ReturnType<DecisionRouter['nextBoundedFallback']> | ReturnType<DecisionRouter['nextUnrestrictedFallback']>): RuntimeResult {
    if (!decision) throw new RuntimeInvariantError('No routing decision selected')

    if (decision.source === 'dna_check' && decision.checkOutcome === 'prerequisite_required') {
      return { source: decision.source, action: decision.action, checkOutcome: 'prerequisite_required', output: { resolutionRequired: true } }
    }

    if (decision.source === 'dna_check' && decision.checkOutcome === 'check_fail') {
      return { source: decision.source, action: decision.action, checkOutcome: 'check_fail', prevented: true }
    }

    if (decision.source === 'db_execute' && decision.operator) {
      return {
        source: decision.source,
        action: decision.action,
        output: this.db.resolve(input)?.execute(input),
        fallbackReason: decision.fallbackReason,
      }
    }

    return {
      source: decision.source,
      action: decision.action,
      prevented: Boolean(decision.prevented),
      checkOutcome: decision.checkOutcome,
      fallbackReason: decision.fallbackReason,
    }
  }

  private pickFallbackRoute(input: string, sessionId: string, forceDeclared: boolean) {
    if (forceDeclared) return this.router.nextUnrestrictedFallback(true)

    const bounded = this.router.nextBoundedFallback()
    const boundedResult = this.simulateTurboAssist(input, sessionId)
    if (boundedResult !== undefined) return bounded

    return this.router.nextUnrestrictedFallback(false)
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
}

export function createRuntimeCore(policy: RuntimePolicy) {
  const dna = new DNARuntime()
  const db = new MumpixDbAdapter()
  const execution = new ExecutionRegistry()
  const idempotency = new IdempotencyRegistry(policy.idempotency)
  const tasks = new TaskRegistry()
  return new RuntimeCore(dna, db, execution, idempotency, tasks, policy)
}
