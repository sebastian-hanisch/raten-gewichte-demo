"""Worst-Case am Slider-Maximum und an den Live-Budgets (DEMO-PLAYBOOK §1c): Hänger und Aufblähen im eigenen Prozess mit Zeitgrenze fangen; dazu dieselbe Rechnung unter zwei
Hash-Seeds (Reihenfolge von Mengen und Dictionaries darf nichts ändern, §1d). Keine Zeit-Assertion, nur ein Timeout als Netz."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = """
import sys
sys.path.insert(0, {root!r})
import rgw_constants as C, rgw_evaluation as V, rgw_experiments as X
for task in C.TASK_KEYS:
    a = V.analyse(V.Settings(task=task, m=C.M_MAX, sigma=C.SIGMA_MAX, r=C.R_MAX))
g = X.growth_live('zweipunkt', max(C.LIVE_PATH_T), C.LIVE_PATH_BUDGET // max(C.LIVE_PATH_T), 0.5, 1)
assert 0.9 < g['below'] <= 1.0
g = X.growth_live('lognormal', max(C.LIVE_PATH_T), max(C.LIVE_PATH_N), C.SIGMA_MAX, 1)
b = X.back_live('pareto', 2.2, max(X.E3_NS), C.LIVE_BACK_BUDGET // max(X.E3_NS), 1)
assert b['hm_rmse']['mean'] < b['am_rmse']['mean']
r = X.reversal_live(max(X.E4_LAYERS), C.LIVE_TABLE_BUDGET // (4 * max(X.E4_LAYERS)), 1)
assert r['share']['mean'] < 0.001
c = X.coupling_cell(0.7, max(X.E5_SIGMA_DS), C.LIVE_COMPARISONS, seeds=(1,))
assert c['b_ahead_pooled']['mean'] > 0.7
x = X.cost_cell('faktoren', 1.0, C.M_MAX, 500, seeds=(1,))
assert x['abs']['geometrisch']['mean'] < 1e-9
print('ok')
"""

HASH_CODE = """
import json, sys
sys.path.insert(0, {root!r})
import rgw_evaluation as V, rgw_experiments as X
a = V.analyse(V.Settings(task='strecke', m=30, sigma=0.7, r=0.5, seed=11))
l = V.analyse(V.Settings(task='lose', m=50, sigma=1.0, seed=11))
out = {{'means': a.means, 'truth': a.truth, 'lots': [l.arithmetic, l.harmonic],
        'cost': X.cost_cell('zeit', 0.5, 10, 100, seeds=(5,))['abs'], 'back': X.back_cell('geometrisch', 0.3, 20, 50, seeds=(5,))['hm_bias'],
        'weights': X.weight_cell(3, 1.0, 2000, seeds=(5,))['share'], 'growth': X.growth_live('lognormal', 20, 500, 0.4, 5)['below']}}
print(json.dumps(out, sort_keys=True))
"""


def _run(code, hash_seed=None, timeout=180):
    env = None
    if hash_seed is not None:
        import os
        env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    return subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code.format(root=str(ROOT))], capture_output=True, text=True, timeout=timeout, env=env)


def test_slider_maximum_and_live_budgets_finish_in_their_own_process_within_the_time_limit():
    done = _run(CODE)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-800:]


def test_the_same_scenario_gives_the_same_numbers_under_two_hash_seeds():
    one, two = _run(HASH_CODE, 1), _run(HASH_CODE, 2)
    assert one.returncode == 0 and two.returncode == 0, (one.stderr[-400:], two.stderr[-400:])
    assert one.stdout == two.stdout and one.stdout.strip().startswith("{")
