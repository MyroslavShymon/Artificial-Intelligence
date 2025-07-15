import time
import heapq
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tabulate import tabulate

class Task:
    def __init__(self, n=8):
        self.n = n
        self.initial_state = [random.randint(0, n - 1) for _ in range(n)]

    def is_goal(self, state):
        return self.heuristic(state) == 0

    def successors(self, state):
        successors = []
        for col in range(self.n):
            for row in range(self.n):
                if state[col] != row:
                    new_state = state[:]
                    new_state[col] = row
                    successors.append(new_state)
        return successors

    def heuristic(self, state):
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

def ldfs(task, depth_limit):
    def recursive_dls(state, depth):
        if task.is_goal(state):
            return state, 1, 1
        if depth == 0:
            return None, 0, 0
        generated = 0
        stored = 1
        for successor in task.successors(state):
            result, gen, sto = recursive_dls(successor, depth - 1)
            generated += gen + 1
            stored += sto
            if result is not None:
                return result, generated, stored
        return None, generated, stored

    return recursive_dls(task.initial_state, depth_limit)

def a_star(task):
    initial_state = task.initial_state
    frontier = [(task.heuristic(initial_state), 0, initial_state)]
    explored = set()
    generated = 0
    stored = 1

    while frontier:
        _, cost, state = heapq.heappop(frontier)
        if tuple(state) in explored:
            continue
        explored.add(tuple(state))

        if task.is_goal(state):
            return state, generated, stored

        for successor in task.successors(state):
            generated += 1
            heapq.heappush(frontier, (cost + 1 + task.heuristic(successor), cost + 1, successor))
            stored = max(stored, len(frontier))
    return None, generated, stored

def run_with_timeout(func, *args, timeout=10):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print("Час виконання перевищено")
            return None, 0, 0

def run_experiments(task_class, algorithm, repetitions=20):
    results = []
    for exp_id in range(1, repetitions + 1):
        task = task_class()
        start_time = time.time()

        if algorithm == 'LDFS':
            result, generated, stored = run_with_timeout(ldfs, task, task.n * 2, timeout=10)
        elif algorithm == 'A*':
            result, generated, stored = run_with_timeout(a_star, task, timeout=10)
        else:
            raise ValueError("Unknown algorithm")

        elapsed_time = time.time() - start_time
        results.append({
            'Експеримент': exp_id,
            'К-сть ферзів, що треба переставити': 1,
            'Вихідний стан': ''.join(map(str, task.initial_state)),
            'Алгоритм': algorithm,
            'Цільовий стан': ''.join(map(str, result)) if result else 'None',
            'Час,с': round(elapsed_time, 3),
            'Кількість згенерованих вузлів': generated,
            'Максимальна кількість вузлів, що одночасно зберігалися в пам\'яті': stored
        })
        print(f"Експеримент {exp_id} завершено: {results[-1]}")
    return results

if __name__ == "__main__":
    task_class = lambda: Task()
    all_results = []

    print("Запуск LDFS:")
    ldfs_results = run_experiments(task_class, algorithm='LDFS', repetitions=20)
    all_results.extend(ldfs_results)

    print("Запуск A*:")
    a_star_results = run_experiments(task_class, algorithm='A*', repetitions=20)
    all_results.extend(a_star_results)

    df = pd.DataFrame(all_results)
    # Генеруємо таблицю з використанням tabulate
    table = tabulate(
        df,
        headers="keys",
        tablefmt="pretty",
        showindex=False,
        numalign="right",
        stralign="left"
    )
    print("\nОстаточні результати у вигляді таблиці:\n")
    print(table)

    input("Натисніть Enter для завершення програми...")
