"""
Hyperparameter search.

Every method in the paper -- ours and all nine baselines -- is tuned the same
way: coordinate ascent over *every* parameter the method accepts, maximising
F1 on the tuning half.

Coordinate ascent rather than a full product grid: the cost is the SUM of the
level counts instead of their product, which is what makes it affordable to
sweep each method's complete parameter set rather than a hand-picked subset.
"""


def coordinate_ascent(score, grid, passes=3, start=None):
    """
    Maximise `score(config)` over `grid` = {name: [levels]}.

    Starts from `start` (falling back to the first level of every axis), then
    repeatedly sweeps one axis at a time, keeping any single-parameter change
    that improves the objective, until a full pass improves nothing or
    `passes` passes are done.  Ties keep the incumbent, so a parameter with no
    effect leaves the configuration untouched.

    `score` returns a float, or None for a configuration the method rejects.
    Returns (best_config, best_score, n_evaluations).
    """
    if not grid:
        return {}, score({}), 1
    start = start or {}
    cur = {k: (start[k] if k in start else grid[k][0]) for k in grid}
    best = score(cur)
    n = 1
    if best is None:
        cur = {k: grid[k][0] for k in grid}
        best = score(cur)
        n += 1
        if best is None:
            raise ValueError('no valid configuration in the grid')

    for _ in range(passes):
        improved = False
        for axis in grid:
            for level in grid[axis]:
                if level == cur[axis]:
                    continue
                trial = dict(cur)
                trial[axis] = level
                s = score(trial)
                n += 1
                if s is not None and s > best + 1e-12:
                    best, cur, improved = s, trial, True
        if not improved:
            break
    return cur, best, n
