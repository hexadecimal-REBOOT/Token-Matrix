import { OperatorGapEvent } from '../shared/types'

export type OperatorSuggestion = {
  suggestedName: string
  intent: string
  domain: string
  payloadShape: string[]
  examples: string[]
}

export class OperatorSuggester {
  suggestOperator(input: { intent: string; domain: string; payloadShape: string[]; examples: string[] }): OperatorSuggestion {
    const normalized = input.intent.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'unknown'
    return {
      suggestedName: `${input.domain}.${normalized}`,
      intent: input.intent,
      domain: input.domain,
      payloadShape: input.payloadShape,
      examples: input.examples,
    }
  }

  fromGap(event: OperatorGapEvent, examples: string[] = [event.intent]): OperatorSuggestion {
    return this.suggestOperator({
      intent: event.intent,
      domain: event.domain,
      payloadShape: event.payloadShape,
      examples,
    })
  }
}
