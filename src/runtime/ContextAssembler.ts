import { SessionState } from './SessionRuntime'

export class ContextAssembler {
  assemble(session: SessionState, input: string): string {
    const recent = session.history.slice(-6).join('\n')
    return `Recent context:\n${recent}\n\nInput:\n${input}`
  }
}
