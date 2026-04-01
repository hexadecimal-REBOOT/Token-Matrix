import { nextId } from '../shared/ids'
import { TaskStateBase } from '../shared/types'

export class TaskRegistry {
  private readonly tasks = new Map<string, TaskStateBase>()

  register(task: Omit<TaskStateBase, 'id' | 'createdAt' | 'status'> & Partial<Pick<TaskStateBase, 'status'>>): string {
    const id = nextId('task')
    this.tasks.set(id, {
      ...task,
      id,
      status: task.status ?? 'pending',
      createdAt: Date.now(),
      notified: false,
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

  fail(taskId: string, _error?: string): void {
    this.update(taskId, (task) => ({ ...task, status: 'failed', endTime: Date.now() }))
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
}
