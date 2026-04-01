let counter = 0

export function nextId(prefix: string): string {
  counter += 1
  return `${prefix}_${Date.now()}_${counter}`
}
