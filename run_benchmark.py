"""Headless runner for the preserved coursework simulation."""
import argparse
import csv
import random
from grid_simulation import build_scaling_config, run_episode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results_all.csv')
    args = parser.parse_args()
    if args.grid_size < 5 or args.episodes < 1:
        parser.error('grid-size must be at least 5 and episodes at least 1')
    scaling = build_scaling_config(args.grid_size)
    results = []
    for config in ['A_star_only', 'A_star_minimax', 'optimized_minimax']:
        for episode in range(1, args.episodes + 1):
            random.seed(args.seed + episode)
            results.append(run_episode(config, episode, scaling, render=False, render_sleep_sec=0))
    with open(args.output, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved {len(results)} episodes to {args.output}')

if __name__ == '__main__':
    main()
