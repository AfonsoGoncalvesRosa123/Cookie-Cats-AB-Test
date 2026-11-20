"""
Cookie Cats A/B test: should the first gate move from level 30 to level 40?

Control   = gate_30 (current game)
Treatment = gate_40 (gate moved later)

Run:  python ab_test.py            (expects data/cookie_cats.csv)
      python ab_test.py --data path/to/cookie_cats.csv

Outputs:
  results/summary.json   every number used in the README
  figures/*.png          charts used in the README

Statistics are written out with numpy/scipy (no black-box A/B library),
so every formula is visible in this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONTROL, TREATMENT = "gate_30", "gate_40"
METRICS = ["retention_1", "retention_7"]
ALPHA = 0.05
POWER = 0.80
N_BOOT = 10_000
N_POSTERIOR = 200_000
SEED = 42

COLORS = {CONTROL: "#1f4e79", TREATMENT: "#e07b39"}
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


# --------------------------------------------------------------------------
# 1. Load and sanity checks
# --------------------------------------------------------------------------
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
    for m in METRICS:
        # Kaggle file stores True/False; make sure they are 0/1 integers
        df[m] = df[m].astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
        if df[m].isna().any():
            raise SystemExit(f"Unexpected values in column {m}")
        df[m] = df[m].astype(int)
    df["version"] = df["version"].str.strip()
    unknown = set(df["version"]) - {CONTROL, TREATMENT}
    if unknown:
        raise SystemExit(f"Unknown groups in 'version': {unknown}")
    return df


def sanity_checks(df: pd.DataFrame) -> dict:
    n = df.groupby("version").size()
    n_c, n_t = int(n[CONTROL]), int(n[TREATMENT])
    # Sample-ratio mismatch: the test was designed as a 50/50 split
    chi2, p_srm = stats.chisquare([n_c, n_t], f_exp=[(n_c + n_t) / 2] * 2)
    rounds = df["sum_gamerounds"]
    top = df.nlargest(3, "sum_gamerounds")[["userid", "version", "sum_gamerounds"]]
    return {
        "rows": int(len(df)),
        "duplicate_userids": int(df["userid"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "n_control": n_c,
        "n_treatment": n_t,
        "share_control": n_c / (n_c + n_t),
        "srm_chi2": float(chi2),
        "srm_p_value": float(p_srm),
        "zero_round_players": int((rounds == 0).sum()),
        "zero_round_share": float((rounds == 0).mean()),
        "gamerounds_p99": float(rounds.quantile(0.99)),
        "gamerounds_max": int(rounds.max()),
        "top_3_players": top.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# 2. Frequentist test for a difference in proportions
# --------------------------------------------------------------------------
def two_proportion_test(x_c: int, n_c: int, x_t: int, n_t: int) -> dict:
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    # p-value: pooled standard error (H0 says both groups share one rate)
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    # confidence interval: unpooled standard error (no H0 assumed)
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    return {
        "rate_control": p_c,
        "rate_treatment": p_t,
        "diff": diff,
        "relative_lift": diff / p_c,
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "z": float(z),
        "p_value": float(p_value),
    }


def bootstrap_diff(x_c: int, n_c: int, x_t: int, n_t: int, rng) -> np.ndarray:
    # Resampling n 0/1 values with replacement and taking the mean is the same
    # as drawing Binomial(n, p_hat) / n, so this is an exact, fast bootstrap.
    boot_c = rng.binomial(n_c, x_c / n_c, N_BOOT) / n_c
    boot_t = rng.binomial(n_t, x_t / n_t, N_BOOT) / n_t
    return boot_t - boot_c


def srm_stress_test(x_c: int, n_c: int, x_t: int, n_t: int) -> dict:
    """Sample-ratio-mismatch stress test. Drop the extra players of the bigger
    group in the way that makes treatment look as good as possible, then test
    again. If treatment still does not win, the SRM cannot flip the decision."""
    excess = abs(n_t - n_c)
    if n_t > n_c:   # drop extra treatment players who did NOT return
        x_t2, n_t2, x_c2, n_c2 = x_t, n_t - excess, x_c, n_c
    else:           # drop extra control players who DID return
        x_t2, n_t2, x_c2, n_c2 = x_t, n_t, x_c - excess, n_c - excess
    return {"excess_players_removed": excess,
            **two_proportion_test(x_c2, n_c2, x_t2, n_t2)}


# --------------------------------------------------------------------------
# 3. Power: what could this test detect, and what would it need?
# --------------------------------------------------------------------------
def required_n_per_group(p1: float, p2: float) -> int:
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    z_b = stats.norm.ppf(POWER)
    p_bar = (p1 + p2) / 2
    num = (
        z_a * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def minimum_detectable_effect(p: float, n_c: int, n_t: int) -> float:
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    z_b = stats.norm.ppf(POWER)
    return float((z_a + z_b) * np.sqrt(p * (1 - p) * (1 / n_c + 1 / n_t)))


def achieved_power(p_c: float, p_t: float, n_c: int, n_t: int) -> float:
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    shift = abs(p_t - p_c) / se
    return float(stats.norm.sf(z_a - shift) + stats.norm.cdf(-z_a - shift))


# --------------------------------------------------------------------------
# 4. Bayesian view: probability that gate_30 is really better
# --------------------------------------------------------------------------
def bayesian(x_c: int, n_c: int, x_t: int, n_t: int, rng) -> dict:
    # Beta(1, 1) prior -> Beta(1 + successes, 1 + failures) posterior
    post_c = rng.beta(1 + x_c, 1 + n_c - x_c, N_POSTERIOR)
    post_t = rng.beta(1 + x_t, 1 + n_t - x_t, N_POSTERIOR)
    diff = post_t - post_c
    return {
        "prob_control_better": float((post_c > post_t).mean()),
        "prob_treatment_better": float((post_t > post_c).mean()),
        # expected retention lost (in proportion points) if we pick that arm
        "expected_loss_if_ship_treatment": float(np.maximum(post_c - post_t, 0).mean()),
        "expected_loss_if_keep_control": float(np.maximum(post_t - post_c, 0).mean()),
        "credible_low": float(np.quantile(diff, ALPHA / 2)),
        "credible_high": float(np.quantile(diff, 1 - ALPHA / 2)),
    }


# --------------------------------------------------------------------------
# 5. Engagement (game rounds): heavy tail, so rank test + medians
# --------------------------------------------------------------------------
def gamerounds_test(df: pd.DataFrame, rng) -> dict:
    c = df.loc[df.version == CONTROL, "sum_gamerounds"].to_numpy()
    t = df.loc[df.version == TREATMENT, "sum_gamerounds"].to_numpy()
    u, p = stats.mannwhitneyu(t, c, alternative="two-sided")
    boot_med = np.empty(2000)
    for i in range(boot_med.size):
        boot_med[i] = np.median(rng.choice(t, t.size)) - np.median(rng.choice(c, c.size))
    return {
        "mean_control": float(c.mean()),
        "mean_treatment": float(t.mean()),
        "median_control": float(np.median(c)),
        "median_treatment": float(np.median(t)),
        "median_diff_ci_low": float(np.quantile(boot_med, ALPHA / 2)),
        "median_diff_ci_high": float(np.quantile(boot_med, 1 - ALPHA / 2)),
        "mann_whitney_u": float(u),
        "mann_whitney_p": float(p),
        # P(random treatment player plays more rounds than random control player)
        "prob_superiority": float(u / (c.size * t.size)),
    }


# --------------------------------------------------------------------------
# 6. Segments (exploratory only)
# --------------------------------------------------------------------------
SEGMENTS = [(0, 29, "0-29 rounds"), (30, 39, "30-39 rounds"),
            (40, 99, "40-99 rounds"), (100, np.inf, "100+ rounds")]


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def segment_analysis(df: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi, label in SEGMENTS:
        seg = df[(df.sum_gamerounds >= lo) & (df.sum_gamerounds <= hi)]
        g = seg.groupby("version")["retention_7"].agg(["sum", "count"])
        if CONTROL not in g.index or TREATMENT not in g.index:
            continue
        r = two_proportion_test(int(g.loc[CONTROL, "sum"]), int(g.loc[CONTROL, "count"]),
                                int(g.loc[TREATMENT, "sum"]), int(g.loc[TREATMENT, "count"]))
        rows.append({"segment": label,
                     "n_control": int(g.loc[CONTROL, "count"]),
                     "n_treatment": int(g.loc[TREATMENT, "count"]), **r})
    for row, adj in zip(rows, holm([r["p_value"] for r in rows])):
        row["p_value_holm"] = adj
    return rows


# --------------------------------------------------------------------------
# 7. Charts
# --------------------------------------------------------------------------
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_retention(results: dict, counts: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    z = stats.norm.ppf(1 - ALPHA / 2)
    for ax, m in zip(axes, METRICS):
        r = results[m]
        rates = [r["rate_control"], r["rate_treatment"]]
        ns = [counts[CONTROL], counts[TREATMENT]]
        errs = [z * np.sqrt(p * (1 - p) / n) for p, n in zip(rates, ns)]
        bars = ax.bar([CONTROL, TREATMENT], [p * 100 for p in rates],
                      yerr=[e * 100 for e in errs], capsize=6,
                      color=[COLORS[CONTROL], COLORS[TREATMENT]], width=0.55)
        for b, p in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, p * 100 / 2, f"{p:.1%}",
                    ha="center", color="white", fontweight="bold")
        day = m.split("_")[1]
        ax.set_title(f"Day-{day} retention (95% CI)")
        ax.set_ylabel("Players returning (%)")
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retention_by_group.png", dpi=150)
    plt.close(fig)


def plot_diff_ci(results: dict):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    labels = ["Day-1 retention", "Day-7 retention"]
    for i, m in enumerate(METRICS):
        r = results[m]
        color = "#c0392b" if r["ci_high"] < 0 else ("#27ae60" if r["ci_low"] > 0 else "#7f8c8d")
        ax.errorbar(r["diff"] * 100, i,
                    xerr=[[(r["diff"] - r["ci_low"]) * 100], [(r["ci_high"] - r["diff"]) * 100]],
                    fmt="o", color=color, capsize=6, markersize=8, lw=2)
        ax.text(r["ci_high"] * 100 + 0.15, i,
                f"{r['diff'] * 100:+.2f} pp  (p = {r['p_value']:.3f})", va="center")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(METRICS)), labels)
    ax.set_ylim(-0.6, len(METRICS) - 0.4)
    ax.set_xlabel("gate_40 minus gate_30 (percentage points), 95% CI")
    ax.set_title("Effect of moving the gate to level 40")
    ax.spines[["top", "right"]].set_visible(False)
    lo = min(results[m]["ci_low"] for m in METRICS) * 100
    hi = max(results[m]["ci_high"] for m in METRICS) * 100
    ax.set_xlim(min(lo, 0) - 0.4, max(hi, 0) + 2.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "effect_ci.png", dpi=150)
    plt.close(fig)


def plot_bootstrap(boots: dict, results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, METRICS):
        d = boots[m] * 100
        ax.hist(d, bins=60, color=COLORS[TREATMENT], alpha=0.85)
        lo, hi = np.quantile(d, [ALPHA / 2, 1 - ALPHA / 2])
        ax.axvspan(lo, hi, color="grey", alpha=0.15, label="95% bootstrap interval")
        ax.axvline(0, color="black", lw=1)
        share_below = (boots[m] < 0).mean()
        ax.set_title(f"{m.replace('_', ' ').title()}: {share_below:.1%} of resamples < 0")
        ax.set_xlabel("gate_40 minus gate_30 (pp)")
        style(ax)
    axes[0].set_ylabel("Bootstrap resamples")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, frameon=False, loc="lower center")
    fig.savefig(FIG_DIR / "bootstrap_distribution.png", dpi=150)
    plt.close(fig)


def plot_gamerounds(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.arange(0, 101, 2)
    for g in (CONTROL, TREATMENT):
        r = df.loc[df.version == g, "sum_gamerounds"]
        # weights -> share of ALL players in the group (players above 100 not drawn)
        ax.hist(r[r < 100], bins=bins, histtype="step", lw=2, color=COLORS[g],
                weights=np.full((r < 100).sum(), 1 / len(r)), label=g)
    for level in (30, 40):
        ax.axvline(level, color=COLORS[CONTROL if level == 30 else TREATMENT], ls="--", lw=1)
    ax.set_xlabel("Game rounds in first 14 days (0-99 shown)")
    ax.set_ylabel("Share of players")
    ax.set_title("Where players stop playing")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gamerounds_distribution.png", dpi=150)
    plt.close(fig)


def plot_segments(segments: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, s in enumerate(segments):
        ax.errorbar(s["diff"] * 100, i,
                    xerr=[[(s["diff"] - s["ci_low"]) * 100], [(s["ci_high"] - s["diff"]) * 100]],
                    fmt="o", color="#34495e", capsize=5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(segments)), [s["segment"] for s in segments])
    ax.invert_yaxis()
    ax.set_xlabel("Day-7 retention, gate_40 minus gate_30 (pp), 95% CI")
    ax.set_title("Exploratory: effect by engagement level (not causal, see README)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "segments_day7.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 8. Decision rule written before looking at results
# --------------------------------------------------------------------------
def decide(results: dict, bayes: dict) -> str:
    r7 = results["retention_7"]
    if r7["ci_low"] > 0:
        return "SHIP gate_40: day-7 retention is significantly higher."
    if r7["ci_high"] < 0:
        return ("KEEP gate_30: moving the gate to level 40 significantly lowers "
                "day-7 retention, the metric tied to long-term revenue.")
    if bayes["retention_7"]["prob_control_better"] > 0.95:
        return ("KEEP gate_30: not significant at 5%, but gate_30 is better with "
                f"{bayes['retention_7']['prob_control_better']:.0%} probability "
                "and gate_40 brings no upside.")
    return "NO DECISION: effect too small to detect; keep gate_30 (no reason to change)."


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cookie_cats.csv")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}\n"
                         "Download cookie_cats.csv from Kaggle into the data/ folder.")

    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load(args.data)
    checks = sanity_checks(df)
    counts = {CONTROL: checks["n_control"], TREATMENT: checks["n_treatment"]}

    results, boots, bayes, power = {}, {}, {}, {}
    for m in METRICS:
        g = df.groupby("version")[m].sum()
        x_c, x_t = int(g[CONTROL]), int(g[TREATMENT])
        n_c, n_t = counts[CONTROL], counts[TREATMENT]
        results[m] = two_proportion_test(x_c, n_c, x_t, n_t)
        boots[m] = bootstrap_diff(x_c, n_c, x_t, n_t, rng)
        results[m]["boot_ci_low"] = float(np.quantile(boots[m], ALPHA / 2))
        results[m]["boot_ci_high"] = float(np.quantile(boots[m], 1 - ALPHA / 2))
        results[m]["boot_share_below_zero"] = float((boots[m] < 0).mean())
        results[m]["srm_worst_case"] = srm_stress_test(x_c, n_c, x_t, n_t)
        bayes[m] = bayesian(x_c, n_c, x_t, n_t, rng)
        p_c, p_t = results[m]["rate_control"], results[m]["rate_treatment"]
        power[m] = {
            "mde_abs": minimum_detectable_effect(p_c, n_c, n_t),
            "mde_relative": minimum_detectable_effect(p_c, n_c, n_t) / p_c,
            "achieved_power_for_observed_diff": achieved_power(p_c, p_t, n_c, n_t),
            "n_per_group_needed_for_observed_diff": required_n_per_group(p_c, p_t),
        }

    rounds = gamerounds_test(df, rng)
    segments = segment_analysis(df)
    decision = decide(results, bayes)

    plot_retention(results, counts)
    plot_diff_ci(results)
    plot_bootstrap(boots, results)
    plot_gamerounds(df)
    plot_segments(segments)

    summary = {"sanity_checks": checks, "retention": results, "bayesian": bayes,
               "power": power, "gamerounds": rounds, "segments_retention_7": segments,
               "decision": decision,
               "settings": {"alpha": ALPHA, "power": POWER, "n_boot": N_BOOT, "seed": SEED}}
    with open(RES_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- console report (ASCII only, safe for Windows PowerShell) ----
    line = "-" * 64
    print(line)
    print("SANITY CHECKS")
    print(f"  rows {checks['rows']:,} | duplicate ids {checks['duplicate_userids']} | "
          f"missing values {checks['missing_values']}")
    print(f"  {CONTROL}: {checks['n_control']:,}  {TREATMENT}: {checks['n_treatment']:,}  "
          f"(control share {checks['share_control']:.2%})")
    print(f"  SRM chi-square = {checks['srm_chi2']:.2f}, p = {checks['srm_p_value']:.4f}")
    print(f"  players with 0 rounds: {checks['zero_round_players']:,} "
          f"({checks['zero_round_share']:.1%}) | max rounds {checks['gamerounds_max']:,} "
          f"| p99 {checks['gamerounds_p99']:.0f}")
    print(line)
    for m in METRICS:
        r, b, pw = results[m], bayes[m], power[m]
        print(m.upper())
        print(f"  {CONTROL} {r['rate_control']:.2%}   {TREATMENT} {r['rate_treatment']:.2%}")
        print(f"  diff {r['diff'] * 100:+.2f} pp  (relative {r['relative_lift']:+.2%})")
        print(f"  95% CI [{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}] pp  "
              f"z = {r['z']:.2f}  p = {r['p_value']:.4f}")
        print(f"  bootstrap 95% CI [{r['boot_ci_low'] * 100:+.2f}, "
              f"{r['boot_ci_high'] * 100:+.2f}] pp")
        w = r["srm_worst_case"]
        print(f"  SRM worst case (drop {w['excess_players_removed']} players): diff "
              f"{w['diff'] * 100:+.2f} pp, CI [{w['ci_low'] * 100:+.2f}, "
              f"{w['ci_high'] * 100:+.2f}], p = {w['p_value']:.4f}")
        print(f"  Bayesian P({CONTROL} better) = {b['prob_control_better']:.1%}")
        print(f"  MDE at 80% power: {pw['mde_abs'] * 100:.2f} pp "
              f"({pw['mde_relative']:.1%} relative)")
        print(f"  power for observed diff: {pw['achieved_power_for_observed_diff']:.0%} | "
              f"n/group needed: {pw['n_per_group_needed_for_observed_diff']:,}")
        print(line)
    print("GAME ROUNDS")
    print(f"  mean {rounds['mean_control']:.1f} vs {rounds['mean_treatment']:.1f} | "
          f"median {rounds['median_control']:.0f} vs {rounds['median_treatment']:.0f}")
    print(f"  Mann-Whitney p = {rounds['mann_whitney_p']:.4f}")
    print(line)
    print("SEGMENTS (day-7 retention, exploratory)")
    for s in segments:
        print(f"  {s['segment']:<13} n={s['n_control'] + s['n_treatment']:>6,}  "
              f"diff {s['diff'] * 100:+.2f} pp  p={s['p_value']:.4f}  "
              f"Holm p={s['p_value_holm']:.4f}")
    print(line)
    print("DECISION:", decision)
    print(line)
    print(f"Saved {RES_DIR / 'summary.json'} and 5 charts in {FIG_DIR}")


if __name__ == "__main__":
    main()
