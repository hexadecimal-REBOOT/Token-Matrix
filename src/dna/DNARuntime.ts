import { CheckOutcome } from '../shared/types'

export type Gene = {
  id: string
  stage: 'block' | 'check' | 'do'
  trigger: RegExp
  pinned?: boolean
  confidence: number
  action: string
  checkOutcome?: CheckOutcome
}

export class DNARuntime {
  constructor(private readonly genes: Gene[] = [], private readonly pinnedFireThreshold = 0.95) {}

  activeGeneSet(): Set<string> {
    return new Set(this.genes.map((g) => g.id))
  }

  match(stage: Gene['stage'], input: string): Gene | undefined {
    for (const gene of this.genes) {
      if (gene.stage !== stage) continue
      const matched = gene.trigger.test(input)
      if (!matched) continue
      if (gene.pinned && gene.confidence < this.pinnedFireThreshold) continue
      if (!gene.pinned && gene.confidence < 0.8) continue
      return gene
    }
    return undefined
  }
}
