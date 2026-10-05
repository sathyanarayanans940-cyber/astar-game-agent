import csv
import heapq
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


Cell = Tuple[int, int]
DEFAULT_GRID_SIZE = 10
OBSTACLE_DENSITY = 0.15
SWAP_INTERVAL_STEPS = 5


@dataclass
class ScalingConfig:
    grid_size: int
    total_cells: int
    obstacle_density: float
    num_obstacles: int
    num_enemies: int
    num_goals: int
    dynamic_obstacle_pairs: int
    max_steps_allowed: int


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def draw(
    agent: Cell,
    enemies: Sequence[Cell],
    goals: Sequence[Cell],
    obstacles: Set[Cell],
    grid_size: int,
) -> None:
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    for x, y in obstacles:
        grid[y][x] = "#"

    for x, y in goals:
        grid[y][x] = "G"

    for x, y in enemies:
        grid[y][x] = "E"

    ax, ay = agent
    grid[ay][ax] = "A"

    for row in grid:
        print(" ".join(row))


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(cell: Cell, grid_size: int) -> List[Cell]:
    x, y = cell
    neighbors: List[Cell] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            neighbors.append((nx, ny))
    return neighbors


def build_scaling_config(grid_size: int) -> ScalingConfig:
    total_cells = grid_size * grid_size
    num_enemies = max(2, grid_size // 8)
    num_obstacles = int(OBSTACLE_DENSITY * total_cells)
    num_goals = max(1, grid_size // 10)
    dynamic_pairs = max(1, grid_size // 20)
    max_steps_allowed = max(50, 2 * total_cells)

    return ScalingConfig(
        grid_size=grid_size,
        total_cells=total_cells,
        obstacle_density=OBSTACLE_DENSITY,
        num_obstacles=num_obstacles,
        num_enemies=num_enemies,
        num_goals=num_goals,
        dynamic_obstacle_pairs=dynamic_pairs,
        max_steps_allowed=max_steps_allowed,
    )


def parse_grid_size_range(default_size: int = DEFAULT_GRID_SIZE) -> Tuple[int, int]:
    try:
        raw_min = input(
            f"Enter minimum grid size N1 (default {default_size}): "
        ).strip()
        raw_max = input(
            f"Enter maximum grid size N2 (default {default_size}): "
        ).strip()
    except EOFError:
        return default_size, default_size

    if not raw_min and not raw_max:
        return default_size, default_size

    try:
        n1 = int(raw_min) if raw_min else default_size
        n2 = int(raw_max) if raw_max else default_size
        if n1 < 2 or n2 < 2:
            raise ValueError
    except ValueError:
        print(
            f"Invalid range input. Falling back to defaults N1={default_size}, N2={default_size}."
        )
        return default_size, default_size

    if n2 < n1:
        n1, n2 = n2, n1
    return n1, n2


def get_heuristic_weights(config_name: str) -> Tuple[float, float, float]:
    if config_name == "A_star_only":
        return 1.0, 0.0, 0.0
    if config_name == "A_star_minimax":
        return 1.0, 0.5, 0.2
    return 1.0, 1.0, 0.35


def generate_environment(
    scaling: ScalingConfig,
) -> Tuple[Cell, List[Cell], List[Cell], Set[Cell], List[Dict[str, object]]]:
    grid_size = scaling.grid_size
    start: Cell = (0, 0)

    all_cells: List[Cell] = [
        (x, y) for y in range(grid_size) for x in range(grid_size)
    ]
    available: List[Cell] = [cell for cell in all_cells if cell != start]
    random.shuffle(available)

    goals = available[: scaling.num_goals]
    available = available[len(goals) :]

    enemies = available[: scaling.num_enemies]
    available = available[len(enemies) :]

    obstacles = set(available[: scaling.num_obstacles])

    reserved = {start, *goals, *enemies}
    free_cells = [
        cell for cell in all_cells if cell not in obstacles and cell not in reserved
    ]

    obstacle_candidates = list(obstacles)
    random.shuffle(obstacle_candidates)
    random.shuffle(free_cells)

    actual_dynamic_pairs = min(
        scaling.dynamic_obstacle_pairs, len(obstacle_candidates), len(free_cells)
    )

    dynamic_pairs: List[Dict[str, object]] = []
    for idx in range(actual_dynamic_pairs):
        dynamic_pairs.append(
            {
                "a": obstacle_candidates[idx],
                "b": free_cells[idx],
                "obstacle_at_a": True,
            }
        )

    return start, goals, enemies, obstacles, dynamic_pairs


def astar(
    start: Cell, goals: Set[Cell], obstacles: Set[Cell], grid_size: int
) -> Tuple[List[Cell], Optional[Cell], Dict[str, float]]:
    stats = {
        "nodes_expanded": 0.0,
        "f_sum": 0.0,
        "g_sum": 0.0,
        "h_sum": 0.0,
        "samples": 0.0,
    }

    if not goals:
        return [], None, stats

    open_list: List[Tuple[int, Cell]] = [(0, start)]
    came: Dict[Cell, Cell] = {}
    g_cost: Dict[Cell, int] = {start: 0}
    visited: Set[Cell] = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if current in visited:
            continue
        visited.add(current)

        current_g = g_cost[current]
        current_h = min(manhattan(current, goal) for goal in goals)
        current_f = current_g + current_h

        stats["nodes_expanded"] += 1
        stats["f_sum"] += current_f
        stats["g_sum"] += current_g
        stats["h_sum"] += current_h
        stats["samples"] += 1

        if current in goals:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path, path[-1], stats

        for neighbor in get_neighbors(current, grid_size):
            if neighbor in obstacles:
                continue

            tentative_g = current_g + 1
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                neighbor_h = min(manhattan(neighbor, goal) for goal in goals)
                neighbor_f = tentative_g + neighbor_h
                heapq.heappush(open_list, (neighbor_f, neighbor))
                came[neighbor] = current

    return [], None, stats


def choose_next_step_with_safety(
    path: Sequence[Cell],
    enemies: Sequence[Cell],
    goals: Sequence[Cell],
    heuristic_weight_w1: float,
    heuristic_weight_w2: float,
    heuristic_weight_w3: float,
) -> Tuple[Optional[Cell], float]:
    if len(path) <= 1:
        return None, 0.0

    candidates = path[1:4]
    best_step = candidates[0]
    best_score = float("-inf")

    for step_rank, step in enumerate(candidates, start=1):
        nearest_enemy = (
            min(manhattan(step, enemy) for enemy in enemies) if enemies else 0
        )
        nearest_goal = min(manhattan(step, goal) for goal in goals) if goals else 0

        # Preserve existing style: short-horizon path progress + safety bonus.
        score = (
            heuristic_weight_w1 * (-step_rank)
            + heuristic_weight_w2 * nearest_enemy
            + heuristic_weight_w3 * (-nearest_goal)
        )

        if score > best_score:
            best_score = score
            best_step = step

    return best_step, best_score


def move_enemies(
    enemies: Sequence[Cell], agent: Cell, obstacles: Set[Cell], grid_size: int
) -> List[Cell]:
    updated: List[Cell] = []
    occupied: Set[Cell] = set()

    for enemy in enemies:
        candidates = [enemy] + get_neighbors(enemy, grid_size)
        valid_candidates = [
            cell for cell in candidates if cell not in obstacles and cell not in occupied
        ]
        if not valid_candidates:
            next_cell = enemy
        else:
            next_cell = min(
                valid_candidates, key=lambda cell: (manhattan(cell, agent), random.random())
            )

        updated.append(next_cell)
        occupied.add(next_cell)

    return updated


def apply_dynamic_swaps(
    dynamic_pairs: List[Dict[str, object]],
    obstacles: Set[Cell],
    agent: Cell,
    enemies: Sequence[Cell],
    goals: Sequence[Cell],
) -> int:
    blocked_cells = {agent, *enemies, *goals}
    swaps_done = 0

    for pair in dynamic_pairs:
        a = pair["a"]
        b = pair["b"]
        obstacle_at_a = pair["obstacle_at_a"]

        obstacle_pos = a if obstacle_at_a else b
        free_pos = b if obstacle_at_a else a

        if free_pos in blocked_cells:
            continue
        if obstacle_pos not in obstacles:
            continue
        if free_pos in obstacles:
            continue

        obstacles.remove(obstacle_pos)
        obstacles.add(free_pos)
        pair["obstacle_at_a"] = not obstacle_at_a
        swaps_done += 1

    return swaps_done


def path_blocked(path: Sequence[Cell], obstacles: Set[Cell]) -> bool:
    return any(step in obstacles for step in path[1:])


def run_episode(
    config_name: str,
    episode_id: int,
    scaling: ScalingConfig,
    render: bool = False,
    render_sleep_sec: float = 0.0,
) -> Dict[str, object]:
    start, goals, enemies, obstacles, dynamic_pairs = generate_environment(scaling)
    goal_id_lookup = {goal: idx for idx, goal in enumerate(goals)}

    heuristic_weight_w1, heuristic_weight_w2, heuristic_weight_w3 = get_heuristic_weights(
        config_name
    )

    initial_num_enemies = len(enemies)
    initial_num_goals = len(goals)
    initial_num_obstacles = len(obstacles)
    initial_dynamic_pairs = len(dynamic_pairs)

    difficulty_index = (
        (
            initial_num_enemies
            + initial_dynamic_pairs
            + scaling.obstacle_density * 100.0
        )
        / scaling.grid_size
    )

    computation_start = time.perf_counter()
    astar_runtime_ms = 0.0
    decision_runtime_ms = 0.0

    success = 0
    goal_reached_id = -1
    capture_step = -1
    evaluation_score_final = 0.0

    total_steps = 0
    actual_path_cost = 0
    optimal_path_cost = -1
    agent_trace: List[Cell] = [start]

    replans_count = 0
    astar_calls = 0
    nodes_expanded = 0
    f_sum = 0.0
    g_sum = 0.0
    h_sum = 0.0
    search_samples = 0.0

    min_enemy_distance = (
        min(manhattan(start, enemy) for enemy in enemies) if enemies else float("inf")
    )
    enemy_distance_samples: List[int] = []
    enemy_encounters_count = 0
    near_collision_count = 0

    total_swaps = 0
    blocked_path_events = 0
    forced_replan_due_to_dynamic = 0

    agent = start
    path: List[Cell] = []
    need_replan = True

    while total_steps < scaling.max_steps_allowed:
        if need_replan or len(path) <= 1:
            astar_calls += 1
            if astar_calls > 1:
                replans_count += 1

            astar_start = time.perf_counter()
            path, _, astar_stats = astar(agent, set(goals), obstacles, scaling.grid_size)
            astar_runtime_ms += (time.perf_counter() - astar_start) * 1000

            nodes_expanded += int(astar_stats["nodes_expanded"])
            f_sum += astar_stats["f_sum"]
            g_sum += astar_stats["g_sum"]
            h_sum += astar_stats["h_sum"]
            search_samples += astar_stats["samples"]

            if astar_calls == 1 and path:
                optimal_path_cost = len(path) - 1

            need_replan = False
            if not path:
                break

        decision_start = time.perf_counter()

        next_step, eval_score = choose_next_step_with_safety(
            path=path,
            enemies=enemies,
            goals=goals,
            heuristic_weight_w1=heuristic_weight_w1,
            heuristic_weight_w2=heuristic_weight_w2,
            heuristic_weight_w3=heuristic_weight_w3,
        )
        evaluation_score_final = eval_score

        if next_step is None:
            decision_runtime_ms += (time.perf_counter() - decision_start) * 1000
            break

        agent = next_step
        total_steps += 1
        agent_trace.append(agent)

        path_index = path.index(next_step)
        path = path[path_index:]

        enemies = move_enemies(enemies, agent, obstacles, scaling.grid_size)

        nearest_enemy = min(manhattan(agent, enemy) for enemy in enemies) if enemies else 0
        min_enemy_distance = min(min_enemy_distance, nearest_enemy)
        enemy_distance_samples.append(nearest_enemy)

        if nearest_enemy <= 2:
            enemy_encounters_count += 1
        if nearest_enemy == 1:
            near_collision_count += 1

        if nearest_enemy == 0:
            capture_step = total_steps
            success = 0
            decision_runtime_ms += (time.perf_counter() - decision_start) * 1000
            break

        if agent in goal_id_lookup:
            success = 1
            goal_reached_id = goal_id_lookup[agent]
            decision_runtime_ms += (time.perf_counter() - decision_start) * 1000
            break

        if total_steps % SWAP_INTERVAL_STEPS == 0 and dynamic_pairs:
            swaps_done = apply_dynamic_swaps(dynamic_pairs, obstacles, agent, enemies, goals)
            total_swaps += swaps_done

            if swaps_done > 0 and path_blocked(path, obstacles):
                blocked_path_events += 1
                forced_replan_due_to_dynamic += 1
                need_replan = True

        if render:
            clear()
            draw(agent, enemies, goals, obstacles, scaling.grid_size)
            if render_sleep_sec > 0:
                time.sleep(render_sleep_sec)

        decision_runtime_ms += (time.perf_counter() - decision_start) * 1000

    actual_path_cost = total_steps
    path_length = len(agent_trace)

    if min_enemy_distance == float("inf"):
        min_enemy_distance = 0

    avg_enemy_distance = (
        sum(enemy_distance_samples) / len(enemy_distance_samples)
        if enemy_distance_samples
        else float(min_enemy_distance)
    )

    path_efficiency = (
        (optimal_path_cost / actual_path_cost)
        if optimal_path_cost >= 0 and actual_path_cost > 0
        else 0.0
    )

    avg_f_value = (f_sum / search_samples) if search_samples else 0.0
    avg_g_value = (g_sum / search_samples) if search_samples else 0.0
    avg_h_value = (h_sum / search_samples) if search_samples else 0.0

    computation_time_ms = (time.perf_counter() - computation_start) * 1000

    return {
        "config": config_name,
        "episode": episode_id,
        "grid_size": scaling.grid_size,
        "total_cells": scaling.total_cells,
        "obstacle_density": scaling.obstacle_density,
        "num_obstacles": initial_num_obstacles,
        "num_enemies": initial_num_enemies,
        "num_goals": initial_num_goals,
        "dynamic_pairs": initial_dynamic_pairs,
        "max_steps_allowed": scaling.max_steps_allowed,
        "success": success,
        "goal_reached_id": goal_reached_id,
        "total_steps": total_steps,
        "path_length": path_length,
        "replans_count": replans_count,
        "nodes_expanded": nodes_expanded,
        "optimal_path_cost": optimal_path_cost,
        "actual_path_cost": actual_path_cost,
        "path_efficiency": round(path_efficiency, 6),
        "evaluation_score_final": round(evaluation_score_final, 6),
        "min_enemy_distance": min_enemy_distance,
        "avg_enemy_distance": round(avg_enemy_distance, 6),
        "enemy_encounters_count": enemy_encounters_count,
        "near_collision_count": near_collision_count,
        "capture_step": capture_step,
        "total_swaps": total_swaps,
        "blocked_path_events": blocked_path_events,
        "forced_replan_due_to_dynamic": forced_replan_due_to_dynamic,
        "avg_f_value": round(avg_f_value, 6),
        "avg_g_value": round(avg_g_value, 6),
        "avg_h_value": round(avg_h_value, 6),
        "heuristic_weight_w1": heuristic_weight_w1,
        "heuristic_weight_w2": heuristic_weight_w2,
        "heuristic_weight_w3": heuristic_weight_w3,
        "computation_time_ms": round(computation_time_ms, 6),
        "astar_runtime_ms": round(astar_runtime_ms, 6),
        "decision_runtime_ms": round(decision_runtime_ms, 6),
        "difficulty_index": round(difficulty_index, 6),
    }


def main() -> None:
    min_grid_size, max_grid_size = parse_grid_size_range(DEFAULT_GRID_SIZE)

    all_results: List[Dict[str, object]] = []
    configs = ["A_star_only", "A_star_minimax", "optimized_minimax"]

    for cfg in configs:
        for ep in range(1, 21):
            grid_size = random.randint(min_grid_size, max_grid_size)
            scaling = build_scaling_config(grid_size)
            result = run_episode(cfg, ep, scaling, render=True, render_sleep_sec=0.15)
            print(cfg, ep, result)
            all_results.append(result)

    if all_results:
        with open("results_all.csv", "w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

    print("Saved results_all.csv")


if __name__ == "__main__":
    main()
