"""Mittel und Aufgaben gegen Handrechnung (Mini-Instanzen in conftest.py), Fehlerfälle und die Zweig-Eigenschaft: Jedes Mittel weicht auf mindestens einer Aufgabe von den anderen ab."""

import math

import numpy as np
import pytest

import rgw_means as M
import rgw_scenario as S
import rgw_tasks as T


def test_speeds_over_equal_distance_give_66_67_not_75(speeds):
    km, hours = T.raw_quantities("strecke", speeds)
    assert km.tolist() == [10.0, 10.0] and hours.tolist() == pytest.approx([0.2, 0.1])
    assert M.total_rate(km, hours) == pytest.approx(20 / 0.3)
    assert M.harmonic(speeds) == pytest.approx(20 / 0.3) == pytest.approx(66.6667, abs=1e-4)
    assert M.arithmetic(speeds) == pytest.approx(75.0)
    assert T.true_total("strecke", speeds) == pytest.approx(66.6667, abs=1e-4)


def test_speeds_over_equal_time_give_75(speeds):
    km, hours = T.raw_quantities("zeit", speeds)
    assert M.total_rate(km, hours) == pytest.approx(150 / 2)
    assert T.true_total("zeit", speeds) == pytest.approx(75.0) == pytest.approx(M.arithmetic(speeds))


def test_the_total_rate_is_the_time_weighted_arithmetic_mean(speeds):
    """Rote Linie: Gesamtgröße = gewichtetes arithmetisches Mittel mit den Nennern als Gewichten ((0,2·50 + 0,1·100)/0,3)."""
    _, hours = T.raw_quantities("strecke", speeds)
    assert M.arithmetic(speeds, hours) == pytest.approx(20 / 0.3)
    assert M.arithmetic(speeds, hours) == pytest.approx(M.harmonic(speeds))


def test_four_means_of_the_factors_by_hand(factors):
    m = M.four_means(factors)
    assert m["arithmetisch"] == pytest.approx(1.05)
    assert m["geometrisch"] == pytest.approx(math.sqrt(0.9)) and m["geometrisch"] == pytest.approx(0.948683, abs=1e-6)
    assert m["harmonisch"] == pytest.approx(2 / (1 / 1.5 + 1 / 0.6)) and m["harmonisch"] == pytest.approx(0.857143, abs=1e-6)
    assert m["quadratisch"] == pytest.approx(math.sqrt(1.305)) and m["quadratisch"] == pytest.approx(1.142366, abs=1e-6)
    assert T.true_total("wechsel", T.alternating_factors(4)) == pytest.approx(0.81 ** 0.25) == pytest.approx(math.sqrt(0.9))


def test_alternating_factors_multiply_to_point_nine_per_double_step():
    f = T.alternating_factors(4)
    assert f.tolist() == [1.5, 0.6, 1.5, 0.6] and float(np.prod(f)) == pytest.approx(0.81)
    assert T.alternating_factors(3).tolist() == [1.5, 0.6, 1.5]
    with pytest.raises(ValueError):
        T.alternating_factors(0)


def test_means_are_ordered_and_equal_on_a_constant_set(factors):
    m = M.four_means(factors)
    assert m["harmonisch"] < m["geometrisch"] < m["arithmetisch"] < m["quadratisch"]
    for value in M.four_means([3.7, 3.7, 3.7]).values():
        assert value == pytest.approx(3.7)


def test_limits_r_minus_infinity_and_plus_infinity_are_min_and_max(factors):
    assert M.power_mean(factors, -np.inf) == 0.6 and M.power_mean(factors, np.inf) == 1.5
    assert M.power_mean(factors, 200.0) == pytest.approx(1.5, rel=0.01) and M.power_mean(factors, -200.0) == pytest.approx(0.6, rel=0.01)


def test_weights_move_the_mean_and_are_normalised(speeds):
    assert M.power_mean(speeds, 1.0, [1.0, 3.0]) == pytest.approx(0.25 * 50 + 0.75 * 100)
    assert M.power_mean(speeds, 1.0, [10.0, 30.0]) == pytest.approx(87.5)
    assert M.power_mean(speeds, 0.0, [1.0, 1.0]) == pytest.approx(math.sqrt(5000))
    assert M.power_mean(speeds, -1.0, [1.0, 1.0]) == pytest.approx(66.6667, abs=1e-4)
    assert M.power_mean(speeds, 1.0, [0.0, 1.0]) == pytest.approx(100.0)       # Gewicht null ist erlaubt


def test_the_mean_works_over_the_last_axis_of_a_stack(speeds):
    stack = np.array([[50.0, 100.0], [10.0, 10.0]])
    assert M.harmonic(stack).tolist() == pytest.approx([66.6667, 10.0], abs=1e-4)
    assert M.power_mean(stack, 1.0).shape == (2,)


@pytest.mark.parametrize("bad", [[], [1.0, 0.0], [1.0, -2.0], [1.0, np.nan], [1.0, np.inf]])
def test_non_positive_or_empty_input_is_an_error(bad):
    with pytest.raises(ValueError):
        M.power_mean(bad, 1.0)


def test_bad_weights_are_an_error(speeds):
    with pytest.raises(ValueError):
        M.power_mean(speeds, 1.0, [-1.0, 2.0])
    with pytest.raises(ValueError):
        M.power_mean(speeds, 1.0, [0.0, 0.0])


def test_huge_exponents_do_not_overflow():
    x = np.array([1.0, 1000.0])
    assert M.power_mean(x, 300.0) == pytest.approx(1000.0 * 0.5 ** (1 / 300), rel=1e-9)
    assert M.power_mean(x, -300.0) == pytest.approx(1.0 * 0.5 ** (-1 / 300), rel=1e-9)


def test_relative_error_has_the_sign_of_too_big(speeds):
    assert M.relative_error(75.0, 66.6667) == pytest.approx(0.125, abs=1e-5)
    assert M.relative_error(60.0, 75.0) == pytest.approx(-0.2)


# ---- Zweig-Tests (Nullspalten-Regel): jedes Mittel ist auf mindestens einer Aufgabe das falsche, jede Aufgabe hat ein anderes richtiges Mittel

@pytest.mark.parametrize("task, wrong", [("strecke", "arithmetisch"), ("strecke", "geometrisch"), ("strecke", "quadratisch"),
                                         ("zeit", "harmonisch"), ("verbrauch", "harmonisch"), ("faktoren", "arithmetisch"), ("faktoren", "harmonisch"), ("wechsel", "arithmetisch")])
def test_the_wrong_mean_misses_the_true_total_on_a_stretched_instance(task, wrong):
    x, truth = T.make_batch(task, S.Draws(3), 1, 6, 0.8)
    miss = abs(M.power_mean(x[0], M.MEAN_R[wrong]) / truth[0] - 1)
    assert miss > 0.02


@pytest.mark.parametrize("key", list(T.RATE_TASKS))
def test_the_right_mean_reproduces_the_true_total(key):
    task = T.RATE_TASKS[key]
    x, truth = T.make_batch(key, S.Draws(5), 50, 12, 0.7)
    assert M.power_mean(x, task.right_r) == pytest.approx(truth, rel=1e-12)


def test_every_mean_is_the_right_one_for_some_task():
    assert {t.right_r for t in T.RATE_TASKS.values()} == {-1.0, 0.0, 1.0}
    assert {t.right_name for t in T.RATE_TASKS.values()} == {"harmonisch", "geometrisch", "arithmetisch"}


def test_consumption_in_km_per_litre_gives_the_wrong_fleet_mean():
    """Zwei Fahrzeuge mit 5 und 10 l/100 km, je 100 km: Gesamtverbrauch 15 l auf 200 km = 7,5 l/100 km (arithmetisch).
    In km/l: 20 und 10, arithmetisch 15 km/l, zurückgerechnet 100/15 = 6,67 l/100 km: falsch (das harmonische Mittel der l/100 km)."""
    c = np.array([5.0, 10.0])
    assert T.true_total("verbrauch", c) == pytest.approx(7.5)
    km_per_l = 100.0 / c
    assert km_per_l.tolist() == [20.0, 10.0]
    back = 100.0 / M.arithmetic(km_per_l)
    assert back == pytest.approx(6.6667, abs=1e-4) and back == pytest.approx(M.harmonic(c))


def test_unknown_task_and_missing_denominator_are_errors():
    with pytest.raises(ValueError):
        T.make_batch("unbekannt", S.Draws(1), 1, 3, 0.5)
    with pytest.raises(ValueError):
        T.raw_quantities("faktoren", np.array([1.0, 2.0]))


def test_scalar_results_are_python_floats_and_batches_are_arrays():
    assert type(M.total_rate([10.0, 10.0], [0.2, 0.1])) is float
    assert type(T.true_total("strecke", np.array([50.0, 100.0]))) is float and type(T.true_total("wechsel", np.array([1.5, 0.6]))) is float
    batch = M.total_rate(np.array([[10.0, 10.0], [20.0, 20.0]]), np.array([[0.2, 0.1], [0.4, 0.2]]))
    assert isinstance(batch, np.ndarray) and batch.shape == (2,)


def test_a_single_period_is_one_up_factor():
    assert T.alternating_factors(1).tolist() == [1.5] and T.alternating_factors(2).tolist() == [1.5, 0.6]
    assert T.true_total("wechsel", T.alternating_factors(1)) == pytest.approx(1.5)
