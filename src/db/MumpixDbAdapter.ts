export type DbOperator = {
  name: string
  canHandle: (input: string) => boolean
  execute: (input: string) => unknown
}

export class MumpixDbAdapter {
  constructor(private readonly operators: DbOperator[] = []) {}

  resolve(input: string): DbOperator | undefined {
    return this.operators.find((op) => op.canHandle(input))
  }
}
