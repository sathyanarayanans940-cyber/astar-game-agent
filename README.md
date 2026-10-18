# A star game agent simulation

Semester 4 AI coursework exploring grid navigation with moving enemies and dynamic obstacles. Implemented in Python with A* search, weighted path selection and CSV instrumentation.

## Approach

Three configurations compare baseline A* and two sets of goal-distance / enemy-distance heuristic weights. The original labels `A_star_minimax` and `optimized_minimax` are retained in code and datasets for traceability. Their decision function scores a short list of candidate path cells; it does not implement recursive Minimax or alpha-beta pruning. Quantum machine learning was part of the report's literature review, not this implementation.

## Run

Python 3.10 or later; the simulator uses only the standard library.

```sh
python3 run_benchmark.py --grid-size 15 --episodes 20 --seed 42
```

This runs 60 episodes without animation and writes `results_all.csv`. For the original interactive terminal visualization, run `python3 grid_simulation.py`.

## Recorded results

`data/` preserves three supplied 60-row datasets: a 15×15 grid, a fixed 5×5 variant, and randomized grid sizes from 15 to 30. Records include success, path length, replans, node expansions, enemy proximity and timing.

These are exploratory coursework results. Historical runs do not use a documented seed, and the configurations were not evaluated on paired environments. Rendering delays can affect recorded timing, and path selection can jump beyond the next adjacent cell. These limitations prevent a reliable performance comparison between configurations.

The headless runner seeds each episode for reproducible reruns and retains the original decision logic, including the limitations above.

## Plotting

```sh
python3 -m pip install -r analysis/requirements.txt
cp data/recorded-grid-15.csv results_all.csv
python3 analysis/generate_graphs.py
python3 analysis/generate_graphs_v2.py
```

The original team coursework is in [AIF1_24](https://github.com/sathyanarayanan2024-bit/AIF1_24). This repository adds documentation and a headless runner. The accompanying report was submitted for the course.
