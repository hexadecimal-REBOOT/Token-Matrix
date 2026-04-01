import { MumpixDbAdapter } from '../db/MumpixDbAdapter'
import { DNARuntime } from '../dna/DNARuntime'
import { CheckOutcome, DecisionSource } from '../shared/types'

export type RoutingDecision = {
  source: DecisionSource
  action: string
  matchedGeneId?: string
  checkOutcome?: CheckOutcome
  operator?: string
  prevented?: boolean
  fallbackReason?: string
}

export class DecisionRouter {
  constructor(private readonly dna: DNARuntime, private readonly db: MumpixDbAdapter) {}

  nextDeterministic(input: string): RoutingDecision | undefined {
    const block = this.dna.match('block', input)
    if (block) {
      return { source: 'dna_block', action: block.action, matchedGeneId: block.id, prevented: true }
    }

    const check = this.dna.match('check', input)
    if (check) {
      const checkOutcome = check.checkOutcome ?? 'check_pass'
      return {
        source: 'dna_check',
        action: check.action,
        matchedGeneId: check.id,
        checkOutcome,
        prevented: checkOutcome === 'check_fail',
      }
    }

    const doGene = this.dna.match('do', input)
    if (doGene) {
      return { source: 'dna_do', action: doGene.action, matchedGeneId: doGene.id }
    }

    const operator = this.db.resolve(input)
    if (operator) {
      return { source: 'db_execute', action: operator.name, operator: operator.name }
    }

    return undefined
  }

  nextBoundedFallback(): RoutingDecision {
    return { source: 'turbo_assist', action: 'turbo_assist', fallbackReason: 'deterministic_exhausted' }
  }

  nextUnrestrictedFallback(force: boolean): RoutingDecision {
    if (force) {
      return { source: 'llm_force_freeform', action: 'llm_force_freeform', fallbackReason: 'force_freeform_exception' }
    }
    return { source: 'llm_freeform', action: 'llm_freeform', fallbackReason: 'bounded_exhausted' }
  }
}
