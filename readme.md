
---

# Texas Hold'em CFR

This repository contains a long-running personal project of mine that explores Texas Hold'em solvers using my personal 'research' build of CFR written in Cython 0.29.28 (Python 3.8.8) for my Macbook Pro M1.

- **Note:** The code is poorly written and maintained.
- **Note:** The code is not optimized for interpretability.

## Things It Does (sort of)
- Blueprint Training (solve positions iteratively to build a solution index)\*
- Counter Factual Regret Minimization
- Self/Human-play in Retro Style
- Currently loaded with a very-VERY rough preflop solve (depth 5, 1000 iterations)
    - i.e. the model risks going hay wire 5 moves after the first non-folding action.
    - **See [results/charts/](results/charts)**


```* = WIP```

## Things to do (maybe)
- [ ] Optimize and debug cfr.fast_forward_gamestate
- [ ] Optimize GameState.betting_history 
    - Dynamic lists of 'objects' is not good
- [ ] Build out LocalManager
    - [ ] Load/Save blueprint based on action space
    - [ ] Perform gamestate abstraction based on the player hash.
    - [x] OPTIMIZE
- [ ] Formalize preflop blueprint construction process
    - Prune depth == Blueprint strategy depth.
    - Blueprints, or strategies, can be chunked and loaded dynamically through the LocalManager. Even abstraction can be handled here.
- [ ] Build out interactive API (for playing/testing)
- [ ] Implement postflop abstractions
    - [ ] See _util.pyx.handtype(...)

- [x] Optimize _utils.dynamic_merge_dicts() 
    - Should still investigate multiple accumulators
- [x] Environment to play the AI
    - [x] Barebones
- [x] Adjust pruning logic to feed back to the global regret and strategy sums. 
## Example


The following is a range chart that displays how often the (UTG) should open raise 1.5x the pot (1.5bb * 1.5 = 2.25bb) in 6-Max Texas Hold'em. Iterations Search depth and gamestate hashing are heavily restricted due to memory limitations and performance constraints. Regardless, the sampled range below reflects a loose version of what you might see on GTO Wizard.

![Preflop UTG Example](results/EX%20Preflop%20UTG%20Open%206%20Max.png)

## Installation
To install the current implementation, please follow the steps below:

**Note:** The current implementation was built on a Macbook Pro M1.

1. **Install Anaconda Package Manager for Python.**

2. **Create the Anaconda Environment:**
    ```sh
    conda create --name Serene python=3.8.8 numpy pandas cython scikit-learn tqdm matplotlib psutil
    ```

3. **Activate the Anaconda Environment:**
    ```sh
    conda activate Serene
    ```

4. **Clone and navigate to source folder:**
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