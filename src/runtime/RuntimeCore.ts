import { MumpixDbAdapter } from '../db/MumpixDbAdapter'
import { DNARuntime } from '../dna/DNARuntime'
import { ExecutionRegistry } from '../execution/ExecutionRegistry'
import { IdempotencyRegistry } from '../idempotency/IdempotencyRegistry'
import { RuntimePolicy } from '../policy/RuntimePolicy'
import { TaskRegistry } from '../tasks/TaskRegistry'
import { nextId } from '../shared/ids'
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
    const session = this.sessionRuntime.getOrCreate(sessionId)
    this.sessionRuntime.pushHistory(sessionId, input)

    const forceDeclared = Boolean(opts.force_freeform)
    if (forceDeclared) this.validateForceFreeform(opts, 'general')

    let decision
    try {
      decision = this.router.route(input)
    } catch {
      decision = this.resolveFallback(input, session.id, opts)
    }

    const scope = opts.scope ?? this.policy.idempotency.getScopeForAction(decision.action)
    const payload = { input, source: decision.source }
    const key = this.idempotency.computeKey({ action: decision.action, payload, scope, sessionId: session.id, taskId: opts.taskId })
    const state = this.idempotency.check(key)

    if (state.status === 'completed' && state.record) {
      return {
        source: decision.source,
        action: decision.action,
        output: state.record.result,
        prevented: state.record.prevented,
        shortCircuited: true,
      }
    }

    if (state.status === 'in_flight') {
      throw new RuntimeInvariantError(`Action already in flight for key ${key}`)
    }

    this.idempotency.start(key, decision.action, scope)

    const result: RuntimeResult = {
      source: decision.source,
      action: decision.action,
      prevented: Boolean(decision.prevented),
      checkOutcome: decision.checkOutcome,
      output: decision.source === 'db_execute' && decision.operator ? this.db.resolve(input)?.execute(input) : undefined,
    }

    this.idempotency.complete(key, result.output, { prevented: result.prevented })

    this.execution.append({
      sessionId,
      taskId: opts.taskId,
      idempotencyKey: key,
      input: { raw: input },
      routing: {
        source: result.source,
        matchedGeneId: decision.matchedGeneId,
        checkOutcome: result.checkOutcome,
        operator: decision.operator,
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
  }

  getSession(sessionId: string) {
    return this.sessionRuntime.get(sessionId)
  }

  async startBackgroundTask(type: string): Promise<string> {
    return this.tasks.register({ type, title: `${type} task` })
  }

  private resolveFallback(input: string, sessionId: string, opts: RuntimeOptions) {
    const force = Boolean(opts.force_freeform)
    if (force) {
      return { source: 'llm_force_freeform' as const, action: 'llm_force_freeform' }
    }

    const assembled = this.contextAssembler.assemble(this.sessionRuntime.getOrCreate(sessionId), input)
    if (assembled.length <= 4000) {
      return { source: 'turbo_assist' as const, action: 'turbo_assist' }
    }

    return { source: 'llm_freeform' as const, action: 'llm_freeform' }
  }

  private validateForceFreeform(opts: RuntimeOptions, domain: string): void {
    if (!opts.forceFreeformReason || !opts.callerSource || !opts.approvedAt) {
      throw new RuntimeInvariantError('force_freeform requires forceFreeformReason, callerSource, and approvedAt')
    }
    if (!this.policy.routing.allowForceFreeform(domain, opts.callerSource)) {
      throw new RuntimeInvariantError(`force_freeform denied for domain ${domain}`)
    }
  }

  replay(action: string): Promise<{ action: string; replayed: boolean }> {
    const hasMatch = this.execution.listBySource('replay').some((r) => r.action.name === action)
    return Promise.resolve({ action, replayed: hasMatch })
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
