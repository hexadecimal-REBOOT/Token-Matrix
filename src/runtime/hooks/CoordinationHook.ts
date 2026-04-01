export type CoordinationEvent =
  | { type: 'task_started'; executionId?: string; taskId: string }
  | { type: 'execution_completed'; executionId: string; recordId: string }

export interface ICoordinationHook {
  publish(event: CoordinationEvent): void
  list(): CoordinationEvent[]
}

export class InMemoryCoordinationHook implements ICoordinationHook {
  private readonly events: CoordinationEvent[] = []

  publish(event: CoordinationEvent): void {
    this.events.push(event)
  }

  list(): CoordinationEvent[] {
    return [...this.events]
  }
}
