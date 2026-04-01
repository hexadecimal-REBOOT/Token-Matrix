import { createRuntimeCore } from '../src/runtime/RuntimeCore.ts'
import { defaultRuntimePolicy } from '../src/policy/RuntimePolicy.ts'

const concurrency = Number(process.argv[2] ?? 1000)
const runtime = createRuntimeCore(defaultRuntimePolicy)

async function main() {
  const tasks = Array.from({ length: concurrency }, (_, i) => runtime.handleInput(`unknown-${i}`))
  await Promise.all(tasks)
  const m = runtime.getMetrics()
  console.log(JSON.stringify({ concurrency, metrics: m }, null, 2))
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
