import { MumpixDbAdapter } from '../db/MumpixDbAdapter'
import { DNARuntime } from '../dna/DNARuntime'
import { DecisionSource, RuntimeInvariantError } from '../shared/types'

export type RoutingDecision = {
  source: DecisionSource
  action: string
  matchedGeneId?: string
  checkOutcome?: 'check_pass' | 'check_fail' | 'prerequisite_required'
  operator?: string
  prevented?: boolean
}

export class DecisionRouter {
  constructor(private readonly dna: DNARuntime, private readonly db: MumpixDbAdapter) {}

  route(input: string): RoutingDecision {
    const block = this.dna.match('block', input)
    if (block) {
      return {
        source: 'dna_block',
        action: block.action,
        matchedGeneId: block.id,
        prevented: true,
      }
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
      return {
        source: 'dna_do',
        action: doGene.action,
        matchedGeneId: doGene.id,
      }
    }

    const operator = this.db.resolve(input)
    if (operator) {
      return {
        source: 'db_execute',
        action: operator.name,
        operator: operator.name,
      }
    }

    throw new RuntimeInvariantError('Deterministic route unresolved')
  }
}
