import { DNARuntime } from '../dna/DNARuntime'

export type SessionState = {
  id: string
  history: string[]
}

export class SessionRuntime {
  private readonly sessions = new Map<string, SessionState>()

  constructor(private readonly dna: DNARuntime) {}

  getOrCreate(sessionId: string): SessionState {
    const existing = this.sessions.get(sessionId)
    if (existing) return existing
    const created = { id: sessionId, history: [] }
    this.sessions.set(sessionId, created)
    return created
  }

  pushHistory(sessionId: string, input: string): void {
    this.getOrCreate(sessionId).history.push(input)
  }

  activeGeneSet(): Set<string> {
    return this.dna.activeGeneSet()
  }

  get(sessionId: string): SessionState | undefined {
    return this.sessions.get(sessionId)
  }
}
