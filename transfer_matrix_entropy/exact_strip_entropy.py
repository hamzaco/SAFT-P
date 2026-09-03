#!/usr/bin/env python3
"""Exact transfer-matrix entropy for patchy particles on infinite strips.

The strip is infinite in x and has open boundaries in y.  A transfer state is
one complete vertical column.  The thermodynamic derivative is evaluated from
left/right Perron eigenvectors; finite differences are used only in tests.

Outputs
-------
data/entropy_table_W3_W4.csv
data/entropy_curves_W1_W4.csv
data/crossovers_W1_W4.csv
data/strong_coupling_W1_W4.csv
data/validation_summary.txt
figures/exact_strip_entropy.pdf
figures/exact_strip_entropy.png
exact_strip_entropy_SI.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TABLE_X = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
WIDTH_COMPARISON_X = (1.0, 2.0, 4.0, 6.0)


@dataclass(frozen=True)
class Model:
    name: str
    labels: tuple[str, ...]
    north: np.ndarray
    east: np.ndarray
    south: np.ndarray
    west: np.ndarray

    @property
    def q(self) -> int:
        return len(self.labels)


STICK = Model(
    name="stick",
    labels=("H", "V"),
    north=np.array((0, 1), dtype=np.int8),
    east=np.array((1, 0), dtype=np.int8),
    south=np.array((0, 1), dtype=np.int8),
    west=np.array((1, 0), dtype=np.int8),
)

LSHAPE = Model(
    name="L-shaped",
    labels=("NE", "ES", "SW", "WN"),
    north=np.array((1, 0, 0, 1), dtype=np.int8),
    east=np.array((1, 1, 0, 0), dtype=np.int8),
    south=np.array((0, 1, 1, 0), dtype=np.int8),
    west=np.array((0, 0, 1, 1), dtype=np.int8),
)

MODELS = (STICK, LSHAPE)


@dataclass
class TransferSystem:
    model: Model
    width: int
    states: np.ndarray
    vertical: np.ndarray
    horizontal: np.ndarray
    bond_weight: np.ndarray
    ground_slope: float
    gauge: np.ndarray
    reduced_weight: np.ndarray

    @classmethod
    def build(cls, model: Model, width: int) -> "TransferSystem":
        if width < 1:
            raise ValueError("width must be positive")
        states = np.asarray(
            list(itertools.product(range(model.q), repeat=width)), dtype=np.int16
        )

        # Row index increases downward.  A vertical bond needs S on the upper
        # site and N on the lower site.
        vertical = np.sum(
            model.south[states[:, :-1]] * model.north[states[:, 1:]], axis=1
        ).astype(np.int16)

        # The first index is the left column and the second the right column.
        horizontal = np.sum(
            model.east[states[:, None, :]]
            * model.west[states[None, :, :]],
            axis=2,
        ).astype(np.int16)

        bond_weight = horizontal.astype(float) + 0.5 * (
            vertical[:, None] + vertical[None, :]
        )

        if model.name == "stick":
            # All-H is the unique maximally bonded column-to-column state.
            ground_slope = float(width)
            gauge = np.zeros(len(states), dtype=float)
        else:
            # Every L orientation has one horizontal and one vertical patch.
            # Horizontal matching contributes at most W/2 per column and the
            # open vertical path contributes floor(W/2).
            ground_slope = width / 2.0 + width // 2
            n_east = np.sum(model.east[states], axis=1)
            gauge = 0.5 * n_east.astype(float)

        # A diagonal similarity transform produces a matrix whose exponents
        # are all <= 0.  It keeps the Perron calculation stable at large x:
        # A = exp(-x*mu) D^{-1} T D, D_CC=exp(x*gauge_C).
        reduced_weight = (
            bond_weight
            - ground_slope
            + gauge[None, :]
            - gauge[:, None]
        )
        if float(np.max(reduced_weight)) > 2.0e-14:
            raise AssertionError("invalid max-plus gauge: positive reduced edge")

        return cls(
            model=model,
            width=width,
            states=states,
            vertical=vertical,
            horizontal=horizontal,
            bond_weight=bond_weight,
            ground_slope=ground_slope,
            gauge=gauge,
            reduced_weight=reduced_weight,
        )

    @property
    def size(self) -> int:
        return len(self.states)

    def balanced_matrix(self, x: float) -> np.ndarray:
        return np.exp(float(x) * self.reduced_weight)

    def perron(self, x: float) -> dict[str, float | np.ndarray]:
        """Return Perron thermodynamics using analytic eigenvector derivatives."""
        a = self.balanced_matrix(x)
        values, left_vectors, right_vectors = eig(
            a, left=True, right=True, check_finite=False
        )
        idx = int(np.argmax(values.real))
        lam_a_complex = values[idx]
        if abs(lam_a_complex.imag) > 2.0e-10 * max(1.0, abs(lam_a_complex.real)):
            raise RuntimeError(f"Perron root unexpectedly complex: {lam_a_complex}")
        lam_a = float(lam_a_complex.real)
        if lam_a <= 0.0:
            raise RuntimeError(f"nonpositive Perron root: {lam_a}")

        left = np.real_if_close(left_vectors[:, idx]).astype(float)
        right = np.real_if_close(right_vectors[:, idx]).astype(float)
        if np.dot(left, right) < 0.0:
            left = -left
        overlap = float(np.dot(left, right))

        # d ln(lambda_A)/dx = l^T[(R o A)]r / (lambda_A l^T r).
        dlog_a = float(
            np.dot(left, (self.reduced_weight * a) @ right)
            / (lam_a * overlap)
        )
        log_lambda = float(x) * self.ground_slope + math.log(lam_a)
        dlog_lambda = self.ground_slope + dlog_a

        # The x*mu terms cancel exactly in the entropy expression.  This form
        # avoids subtracting two large nearly equal numbers at strong coupling.
        entropy = (math.log(lam_a) - float(x) * dlog_a) / self.width
        return {
            "lambda_balanced": lam_a,
            "log_lambda": log_lambda,
            "dlog_lambda": dlog_lambda,
            "entropy": entropy,
            "left": left,
            "right": right,
        }

    def entropy(self, x: float) -> float:
        return float(self.perron(x)["entropy"])

    def ground_matrix(self) -> np.ndarray:
        return np.isclose(self.reduced_weight, 0.0, atol=1.0e-13).astype(float)

    def ground_residual(self) -> tuple[float, float, int]:
        """Return (entropy/particle, spectral radius, number of zero edges)."""
        adjacency = self.ground_matrix()
        values = np.linalg.eigvals(adjacency)
        rho = float(np.max(np.abs(values)))
        residual = math.log(rho) / self.width if rho > 0.0 else -math.inf
        return residual, rho, int(np.sum(adjacency))

    def state_label(self, index: int) -> tuple[str, ...]:
        return tuple(self.model.labels[i] for i in self.states[index])


def direct_counts(
    model: Model, left: Iterable[int], right: Iterable[int] | None = None
) -> int:
    left = tuple(left)
    if right is None:
        return sum(
            int(model.south[left[j]] and model.north[left[j + 1]])
            for j in range(len(left) - 1)
        )
    right = tuple(right)
    return sum(
        int(model.east[a] and model.west[b]) for a, b in zip(left, right)
    )


def brute_periodic_partition(system: TransferSystem, x: float, length: int) -> float:
    """Directly enumerate a short periodic strip, independently of T."""
    z = 0.0
    for columns in itertools.product(range(system.size), repeat=length):
        bonds = 0
        for k, c_idx in enumerate(columns):
            c = system.states[c_idx]
            c_next = system.states[columns[(k + 1) % length]]
            bonds += direct_counts(system.model, c)
            bonds += direct_counts(system.model, c, c_next)
        z += math.exp(x * bonds)
    return z


def validate_system(system: TransferSystem) -> list[str]:
    model = system.model
    width = system.width
    messages: list[str] = []

    if system.size != model.q**width:
        raise AssertionError("wrong number of column states")
    for i, state in enumerate(system.states):
        if int(system.vertical[i]) != direct_counts(model, state):
            raise AssertionError("vertical bond mismatch")
    for i, left in enumerate(system.states):
        for j, right in enumerate(system.states):
            if int(system.horizontal[i, j]) != direct_counts(model, left, right):
                raise AssertionError("horizontal bond mismatch")

    if model.name == "stick":
        if not np.array_equal(system.horizontal, system.horizontal.T):
            raise AssertionError("stick horizontal matrix should be symmetric")
        if not np.array_equal(system.bond_weight, system.bond_weight.T):
            raise AssertionError("stick transfer exponent should be symmetric")
    else:
        # With C explicitly the left column, E(C)-W(C') is directed in state
        # labels.  The requested left/right Perron formula handles this.
        if width >= 1 and np.array_equal(system.horizontal, system.horizontal.T):
            raise AssertionError("L horizontal compatibility should be directed")

    p0 = system.perron(0.0)
    expected_log_lambda = width * math.log(model.q)
    if abs(float(p0["log_lambda"]) - expected_log_lambda) > 2.0e-12:
        raise AssertionError("lambda(0) != q^W")
    if abs(float(p0["entropy"]) - math.log(model.q)) > 2.0e-12:
        raise AssertionError("s(0) != ln(q)")

    # Finite difference is an independent check, not the production derivative.
    x_check = 1.137
    delta = 2.0e-6
    p = system.perron(x_check)
    lp = float(system.perron(x_check + delta)["log_lambda"])
    lm = float(system.perron(x_check - delta)["log_lambda"])
    fd = (lp - lm) / (2.0 * delta)
    analytic = float(p["dlog_lambda"])
    if abs(fd - analytic) > 2.0e-7 * max(1.0, abs(analytic)):
        raise AssertionError(
            f"Perron derivative check failed: analytic={analytic}, FD={fd}"
        )

    expected_vmax = width - 1 if model.name == "stick" else width // 2
    if int(np.max(system.vertical)) != expected_vmax:
        raise AssertionError("unexpected maximum vertical bond count")
    if int(np.max(system.horizontal)) != width:
        raise AssertionError("unexpected maximum horizontal bond count")

    residual, rho, zero_edges = system.ground_residual()
    messages.append(
        f"{model.name:8s} W={width}: N={system.size:3d}, "
        f"v=[{int(np.min(system.vertical))},{int(np.max(system.vertical))}], "
        f"h=[{int(np.min(system.horizontal))},{int(np.max(system.horizontal))}], "
        f"s(0)={float(p0['entropy']):.12f}, "
        f"FD derivative error={abs(fd-analytic):.3e}, "
        f"mu={system.ground_slope:.6g}, rho(A0)={rho:.12g}, "
        f"s_inf={residual:.12f}, zero_edges={zero_edges}"
    )
    return messages


def validate_short_partition(systems: dict[tuple[str, int], TransferSystem]) -> list[str]:
    messages: list[str] = []
    x = 0.731
    length = 3
    for model in MODELS:
        system = systems[(model.name, 2)]
        t = np.exp(x * system.bond_weight)
        transfer_z = float(np.trace(np.linalg.matrix_power(t, length)))
        brute_z = brute_periodic_partition(system, x, length)
        rel = abs(transfer_z - brute_z) / brute_z
        if rel > 2.0e-13:
            raise AssertionError("short-strip direct partition check failed")
        messages.append(
            f"{model.name:8s} W=2, Lx={length}: direct-vs-trace relative error={rel:.3e}"
        )
    return messages


def crossover(system: TransferSystem) -> float:
    target = 0.5 * math.log(system.model.q)
    residual, _, _ = system.ground_residual()
    if residual >= target - 2.0e-13:
        return math.inf
    lo, hi = 0.0, 1.0
    while system.entropy(hi) > target:
        hi *= 2.0
        if hi > 128.0:
            raise RuntimeError("failed to bracket entropy crossover")
    return float(
        brentq(lambda value: system.entropy(value) - target, lo, hi, xtol=2e-13)
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table_rows(systems: dict[tuple[str, int], TransferSystem]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        isolated = math.log(model.q)
        for width in (3, 4):
            system = systems[(model.name, width)]
            for x in TABLE_X:
                p = system.perron(x)
                entropy = float(p["entropy"])
                rows.append(
                    {
                        "model": model.name,
                        "W": width,
                        "x": f"{x:.1f}",
                        "ln_lambda": f"{float(p['log_lambda']):.12f}",
                        "dln_lambda_dx": f"{float(p['dlog_lambda']):.12f}",
                        "s_exact_over_kB": f"{entropy:.12f}",
                        "s_avgE_over_kB": "0.000000000000",
                        "s_isolatedZ_over_kB": f"{isolated:.12f}",
                        "abs_error_avgE": f"{abs(entropy):.12f}",
                        "abs_error_isolatedZ": f"{abs(isolated-entropy):.12f}",
                    }
                )
    return rows




def width_comparison_rows(
    systems: dict[tuple[str, int], TransferSystem]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for x in WIDTH_COMPARISON_X:
            row: dict[str, object] = {"model": model.name, "x": f"{x:.1f}"}
            for width in range(1, 5):
                row[f"s_W{width}_over_kB"] = (
                    f"{systems[(model.name, width)].entropy(x):.12f}"
                )
            rows.append(row)
    return rows


def format_markdown_table(rows: list[dict[str, object]], model: str, width: int) -> str:
    selected = [r for r in rows if r["model"] == model and r["W"] == width]
    lines = [
        "| x | exact s/kB | avgE error | isolated-Z error |",
        "|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['x']} | {float(row['s_exact_over_kB']):.9f} | "
            f"{float(row['abs_error_avgE']):.9f} | "
            f"{float(row['abs_error_isolatedZ']):.9f} |"
        )
    return "\n".join(lines)


def make_plot(
    systems: dict[tuple[str, int], TransferSystem],
    x_grid: np.ndarray,
    curve_data: dict[tuple[str, int], np.ndarray],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    styles = ("-", "--", "-.", (0, (1.5, 1.5)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)

    for ax, model, panel in zip(axes, MODELS, ("a", "b")):
        for width, color, style in zip(range(1, 5), colors, styles):
            ax.plot(
                x_grid,
                curve_data[(model.name, width)],
                color=color,
                linestyle=style,
                linewidth=1.8,
                label=rf"$W={width}$",
            )
        ax.axhline(
            0.0,
            color="0.2",
            linestyle=(0, (5, 2)),
            linewidth=1.1,
            label="avgE",
            zorder=0,
        )
        ax.axhline(
            math.log(model.q),
            color="0.45",
            linestyle=(0, (2, 2)),
            linewidth=1.2,
            label=r"isolated-$Z$",
            zorder=0,
        )
        ax.set_title(f"({panel}) {model.name.capitalize()}")
        ax.set_xlabel(r"$x=\beta\varepsilon_d$")
        ax.set_xlim(0.0, float(x_grid[-1]))
        ax.set_ylim(-0.035, math.log(model.q) + 0.075)
        ax.grid(which="major", color="0.9", linewidth=0.6)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=2.6)
    axes[0].set_ylabel(r"$s/k_{\mathrm{B}}$ per particle")
    fig.tight_layout(w_pad=1.7)
    fig.savefig(output_dir / "exact_strip_entropy.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "exact_strip_entropy.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    table: list[dict[str, object]],
    crossovers: list[dict[str, object]],
    strong: list[dict[str, object]],
    width_comparison: list[dict[str, object]],
) -> None:
    crossover_lookup = {
        (str(r["model"]), int(r["W"])): str(r["x_crossover"]) for r in crossovers
    }
    strong_lookup = {
        (str(r["model"]), int(r["W"])): r for r in strong
    }
    lines = [
        "# Exact orientational entropy on infinite square-lattice strips",
        "",
        "## Transfer-matrix definition",
        "",
        "A state is a complete open vertical column `C=(r1,...,rW)`. The matrix",
        "element is `T_CC' = exp{x[h(C,C')+(v(C)+v(C'))/2]}`. Its Perron root",
        "gives `beta f=-(ln lambda_max)/W`, and the entropy per particle is",
        "`s/kB=[ln lambda_max-x d(ln lambda_max)/dx]/W`. The derivative in every",
        "reported value was evaluated as `l^T(dT/dx)r/(l^T r)`; centered finite",
        "differences were used only as validation.",
        "",
        "For the L particle, the stated rule (left E patch, right W patch) makes",
        "`h(C,C')` directed in the column labels, so `T` is not entrywise symmetric",
        "even after the symmetric split of the vertical energy. Left and right",
        "Perron vectors are therefore both retained. This is the physical",
        "left-to-right compatibility specified in the problem.",
        "",
        "## Numerical results for W=3 and W=4",
        "",
        "The closure entropies are 0 (avgE), ln(2)=0.693147181 (stick isolated-Z),",
        "and ln(4)=1.386294361 (L-shaped isolated-Z). All quantities below are per",
        "particle; the error columns are absolute errors.",
        "",
    ]
    for model in MODELS:
        for width in (3, 4):
            lines.extend(
                [
                    f"### {model.name.capitalize()}, W={width}",
                    "",
                    format_markdown_table(table, model.name, width),
                    "",
                ]
            )

    lines.extend(
        [
            "## Crossovers and strong-coupling limits",
            "",
            "Equal closure error occurs at `s=(ln q)/2`. The exact results are:",
            "",
            "| model | W | crossover x | ground bond slope mu/column | rho(A0) | s(infinity)/kB |",
            "|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for model in MODELS:
        for width in range(1, 5):
            srow = strong_lookup[(model.name, width)]
            lines.append(
                f"| {model.name} | {width} | {crossover_lookup[(model.name, width)]} | "
                f"{float(srow['ground_slope_per_column']):.6f} | "
                f"{float(srow['ground_adjacency_rho']):.9f} | "
                f"{float(srow['s_infinity_over_kB']):.9f} |"
            )

    lines.extend(
        [
            "",
            "## Width comparison at finite coupling",
            "",
            "| model | x | W=1 | W=2 | W=3 | W=4 |",
            "|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for row in width_comparison:
        lines.append(
            f"| {row['model']} | {row['x']} | "
            f"{float(row['s_W1_over_kB']):.9f} | "
            f"{float(row['s_W2_over_kB']):.9f} | "
            f"{float(row['s_W3_over_kB']):.9f} | "
            f"{float(row['s_W4_over_kB']):.9f} |"
        )

    lines.extend(
        [
            "",
            "## SI-ready interpretation",
            "",
            "The transfer result has the required noninteracting limit",
            "`s(0)/kB=ln q` for every width. Increasing directional coupling removes",
            "orientational entropy monotonically. For sticks the all-H strip is the",
            "unique maximally bonded transfer state, so the strong-coupling residual",
            "entropy is zero for W=1-4.",
            "",
            "For L particles, every site carries one horizontal and one vertical",
            "patch. Horizontal dimers contribute at most W/2 bonds per added column,",
            "while a maximum matching of the open vertical path contributes",
            "`floor(W/2)`. Thus the ground-state bond slope is",
            "`mu=W/2+floor(W/2)`. Even widths have a unique vertical perfect-matching",
            "pattern and only a finite set of horizontal dimer phases, giving zero",
            "entropy density. Odd widths retain local freedom in the unpaired",
            "vertical patch: rho(A0)=2 for W=1 and 4 for W=3, hence residual entropies",
            "ln(2) and ln(4)/3 per particle, respectively. More generally, the number",
            "of maximum vertical-patch strings is W+1 for odd W, so",
            "`s(infinity)/kB=ln(W+1)/W` for odd W and zero for even W. The W=1-4",
            "sequence is therefore not monotone at strong coupling, although both the",
            "odd-width envelope and the even-width subsequence converge to zero as W",
            "increases. At finite coupling, sticks show rapid convergence with width;",
            "the L curves show much larger even/odd differences once strong local",
            "bonding sets in.",
            "",
            "The exact entropy always lies between the two closures. avgE is closer",
            "above the reported crossover, whereas isolated-Z is closer below it.",
            "For L-shaped W=1, the exact entropy approaches (ln 4)/2 only as",
            "x -> infinity, so there is no finite crossover.",
            "",
            "## Reproducibility files",
            "",
            "- `exact_strip_entropy.py`: construction, analytic derivatives, tests, tables, and plots",
            "- `data/entropy_table_W3_W4.csv`: requested table including lambda derivative and errors",
            "- `data/entropy_curves_W1_W4.csv`: plotted W=1-4 curves",
            "- `data/width_comparison_W1_W4.csv`: selected finite-x width comparison",
            "- `data/crossovers_W1_W4.csv`: equal-error crossover values",
            "- `data/strong_coupling_W1_W4.csv`: ground-state slopes and residual entropies",
            "- `data/validation_summary.txt`: bond-count, normalization, derivative, and direct partition checks",
            "- `figures/exact_strip_entropy.pdf` and `.png`: publication-quality figure",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="analysis output directory (default: script directory)",
    )
    parser.add_argument(
        "--plot-xmax", type=float, default=8.0, help="largest x shown in the plot"
    )
    parser.add_argument(
        "--plot-points", type=int, default=161, help="number of x points in the plot"
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    systems = {
        (model.name, width): TransferSystem.build(model, width)
        for model in MODELS
        for width in range(1, 5)
    }

    validation_messages: list[str] = []
    for model in MODELS:
        for width in range(1, 5):
            validation_messages.extend(validate_system(systems[(model.name, width)]))
    validation_messages.extend(validate_short_partition(systems))
    (data_dir / "validation_summary.txt").write_text(
        "\n".join(validation_messages) + "\n", encoding="utf-8"
    )

    table = table_rows(systems)
    write_csv(
        data_dir / "entropy_table_W3_W4.csv",
        [
            "model",
            "W",
            "x",
            "ln_lambda",
            "dln_lambda_dx",
            "s_exact_over_kB",
            "s_avgE_over_kB",
            "s_isolatedZ_over_kB",
            "abs_error_avgE",
            "abs_error_isolatedZ",
        ],
        table,
    )

    crossover_rows: list[dict[str, object]] = []
    strong_rows: list[dict[str, object]] = []
    for model in MODELS:
        for width in range(1, 5):
            system = systems[(model.name, width)]
            cross = crossover(system)
            residual, rho, zero_edges = system.ground_residual()
            crossover_rows.append(
                {
                    "model": model.name,
                    "W": width,
                    "target_entropy_over_kB": f"{0.5*math.log(model.q):.12f}",
                    "x_crossover": "infinity" if math.isinf(cross) else f"{cross:.12f}",
                }
            )
            strong_rows.append(
                {
                    "model": model.name,
                    "W": width,
                    "ground_slope_per_column": f"{system.ground_slope:.12f}",
                    "ground_bonds_per_particle": f"{system.ground_slope/width:.12f}",
                    "ground_adjacency_rho": f"{rho:.12f}",
                    "ground_zero_edges": zero_edges,
                    "s_infinity_over_kB": f"{residual:.12f}",
                }
            )
    write_csv(
        data_dir / "crossovers_W1_W4.csv",
        ["model", "W", "target_entropy_over_kB", "x_crossover"],
        crossover_rows,
    )
    write_csv(
        data_dir / "strong_coupling_W1_W4.csv",
        [
            "model",
            "W",
            "ground_slope_per_column",
            "ground_bonds_per_particle",
            "ground_adjacency_rho",
            "ground_zero_edges",
            "s_infinity_over_kB",
        ],
        strong_rows,
    )

    width_rows = width_comparison_rows(systems)
    write_csv(
        data_dir / "width_comparison_W1_W4.csv",
        ["model", "x", "s_W1_over_kB", "s_W2_over_kB", "s_W3_over_kB", "s_W4_over_kB"],
        width_rows,
    )

    x_grid = np.linspace(0.0, args.plot_xmax, args.plot_points)
    curves: dict[tuple[str, int], np.ndarray] = {}
    for model in MODELS:
        for width in range(1, 5):
            system = systems[(model.name, width)]
            curves[(model.name, width)] = np.asarray(
                [system.entropy(float(x)) for x in x_grid]
            )
    curve_table: list[dict[str, object]] = []
    for model in MODELS:
        for width in range(1, 5):
            for x, entropy in zip(x_grid, curves[(model.name, width)]):
                curve_table.append(
                    {
                        "model": model.name,
                        "W": width,
                        "x": f"{x:.8f}",
                        "s_exact_over_kB": f"{entropy:.12f}",
                    }
                )
    write_csv(
        data_dir / "entropy_curves_W1_W4.csv",
        ["model", "W", "x", "s_exact_over_kB"],
        curve_table,
    )
    make_plot(systems, x_grid, curves, figure_dir)
    write_report(
        output_dir / "exact_strip_entropy_SI.md",
        table,
        crossover_rows,
        strong_rows,
        width_rows,
    )

    print("\n".join(validation_messages))
    print(f"Wrote exact strip-entropy analysis to {output_dir}")


if __name__ == "__main__":
    main()
