import { OperatorGapEvent } from '../shared/types'

export interface IOperatorGapRegistry {
  emit(event: OperatorGapEvent): void
  list(): OperatorGapEvent[]
}

export class OperatorGapRegistry implements IOperatorGapRegistry {
  private readonly events: OperatorGapEvent[] = []

  emit(event: OperatorGapEvent): void {
    this.events.push(event)
  }

  list(): OperatorGapEvent[] {
    return [...this.events]
  }
}
