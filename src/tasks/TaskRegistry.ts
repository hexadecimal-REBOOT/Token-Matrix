import { nextId } from '../shared/ids'
import { TaskStateBase } from '../shared/types'

export class TaskRegistry {
  private readonly tasks = new Map<string, TaskStateBase>()

  register(task: Omit<TaskStateBase, 'id' | 'createdAt' | 'notified'> & { notified?: boolean }): string {
    const id = nextId('task')
    this.tasks.set(id, {
      ...task,
      id,
      createdAt: Date.now(),
      notified: task.notified ?? false,
    })
    return id
  }

  update<T extends TaskStateBase>(taskId: string, updater: (task: T) => T): void {
    const task = this.tasks.get(taskId)
    if (!task) throw new Error(`Task not found: ${taskId}`)
    this.tasks.set(taskId, updater(task as T))
  }

  complete(taskId: string): void {
    this.update(taskId, (task) => ({ ...task, status: 'completed', endTime: Date.now() }))
  }

  fail(taskId: string, error?: string): void {
    this.update(taskId, (task) => ({ ...task, status: 'failed', error, endTime: Date.now() }))
  }

  async kill(taskId: string): Promise<void> {
    this.update(taskId, (task) => ({ ...task, status: 'killed', endTime: Date.now() }))
  }

  list(): TaskStateBase[] {
    return [...this.tasks.values()]
  }

  get(taskId: string): TaskStateBase | undefined {
    return this.tasks.get(taskId)
  }

  assertNoDanglingTasks(): void {
    const dangling = this.list().filter((t) => t.status !== 'completed' && t.status !== 'failed' && t.status !== 'killed')
    if (dangling.length > 0) {
      throw new Error(`Spec violation: ${dangling.length} task(s) not in terminal state`)
    }
  }
}
