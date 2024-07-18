
---

# Texas Hold'em CFR

This repository explores Texas Hold'em solvers using a custom build of Counter Factual Regret Minimization (CFR) written in Cython for Macbook Pro M1.

- **Note:** The code is poorly written and maintained.
- **Note:** The code is not optimized for interpretability.

## Features
- Blueprint Training: Solve positions iteratively to build a solution index.
- Counter Factual Regret Minimization (CFR).
- Action space traversal and lookups (cfr.fast_forward_gamestate).
- Self/Human-play in Retro Style (poker_game.play_game).

## Planned Enhancements
- Implement postflop abstractions. (Done, but is based on lossy heuristics)
- Optimize and debug CFR functionalities.
- Log hands for analysis on GTO Wizard.
- Improve GameState and betting history management.
- Refine preflop blueprint construction.
- Refine postflop blueprint construction.
- Build an interactive API for playing/testing.

## Example

The following is a range chart that displays how often the (UTG) should open raise 1.5x the pot (1.5bb * 1.5 = 2.25bb) in 6-Max Texas Hold'em. Iterations search depth and gamestate hashing are restricted due to memory limitations and performance constraints. The sampled range reflects a loose version of what might be seen on GTO Wizard.

![Preflop UTG Example](results/EX%20Preflop%20UTG%20Open%206%20Max.png)

## Installation

To install and run the project, follow these steps:

1. **Install Anaconda Package Manager for Python.**

2. **Create the Anaconda Environment:**
    ```sh
    conda create --name serene python=3.8.8 numpy pandas cython scikit-learn tqdm matplotlib psutil
    ```

3. **Activate the Anaconda Environment:**
    ```sh
    conda activate serene
    ```

4. **Clone and navigate to the source folder:**
    ```sh
    git clone https://github.com/guy477/serene.git
    cd serene/src
    ```

5. **Compile the project:**
    ```sh
    python setup.py build_ext --inplace
    ```

6. **Run the current implementation:**
    ```sh
    python main.py
    ```

---
