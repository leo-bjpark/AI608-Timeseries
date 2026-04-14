from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_synthetic_var_data(T: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate a simple 3-variable stationary process for VAR tutorials."""
    rng = np.random.default_rng(seed)

    e1 = rng.normal(size=T)
    e2 = rng.normal(size=T)
    e3 = rng.normal(size=T)

    y1 = np.zeros(T)
    y2 = np.zeros(T)
    y3 = np.zeros(T)

    for t in range(1, T):
        y1[t] = 0.50 * y1[t - 1] + 0.20 * y2[t - 1] + e1[t]
        y2[t] = 0.10 * y1[t - 1] + 0.40 * y2[t - 1] + 0.10 * y3[t - 1] + e2[t]
        y3[t] = 0.20 * y2[t - 1] + 0.30 * y3[t - 1] + e3[t]

    return pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})


def load_time_series(path: str, sheet_name=0) -> pd.DataFrame:
    """Load time-series data from CSV or Excel based on file suffix."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet_name)

    raise ValueError(f"Unsupported file extension: {suffix}. Use .csv, .xlsx, or .xls")


def prepare_var_data(df: pd.DataFrame, var_names, dropna: bool = True):
    """Select VAR variables, optionally drop NA rows, and return dataframe + float array."""
    var_names = list(var_names)
    missing = [col for col in var_names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input dataframe: {missing}")

    data_df = df[var_names].copy()
    if dropna:
        data_df = data_df.dropna().copy()

    data = data_df.to_numpy(dtype=float)
    return data_df, data


def compute_adf_pvalues(data_df: pd.DataFrame, var_names, autolag: str = "AIC") -> pd.Series:
    """Compute ADF p-values for selected variables."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError("statsmodels is required for ADF checks. Install with `pip install statsmodels`.") from exc

    pvals = {}
    for col in var_names:
        series = data_df[col].dropna()
        if len(series) < 10:
            pvals[col] = np.nan
        else:
            pvals[col] = adfuller(series, autolag=autolag)[1]
    return pd.Series(pvals, name="adf_pvalue")


def apply_var_preprocessing(raw_df: pd.DataFrame, mode: str = "logdiff", enabled: bool = True) -> pd.DataFrame:
    """
    Apply optional preprocessing to VAR input series.
    - enabled=False: return level data
    - enabled=True, mode='logdiff': return log-differenced data
    - enabled=True, mode='level': return level data
    """
    if not enabled or mode == "level":
        return raw_df.copy()
    if mode == "logdiff":
        return np.log(raw_df).diff().dropna()
    raise ValueError(f"Unsupported preprocessing mode: {mode}. Use 'level' or 'logdiff'.")


def summarize_preprocess_on_off(raw_df: pd.DataFrame, var_names, mode: str = "logdiff") -> pd.DataFrame:
    """Return ADF comparison table for preprocessing OFF(level) vs ON(mode)."""
    off_df = apply_var_preprocessing(raw_df, mode=mode, enabled=False)
    on_df = apply_var_preprocessing(raw_df, mode=mode, enabled=True)

    out = pd.DataFrame(
        {
            "adf_off_level": compute_adf_pvalues(off_df, var_names),
            f"adf_on_{mode}": compute_adf_pvalues(on_df, var_names),
        }
    )
    return out


def summarize_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact summary stats used in the tutorial output."""
    return data_df.describe().T[["mean", "std", "min", "max"]]


def plot_raw_series_and_correlation(
    data_df: pd.DataFrame,
    var_names,
    figsize=(12, 4.6),
    series_figsize=None,
    corr_figsize=None,
):
    """Plot raw series and contemporaneous correlation in side-by-side subplots (1x2)."""
    # Backward compatibility: accept legacy args without breaking old notebook cells.
    if series_figsize is not None or corr_figsize is not None:
        pass

    corr = data_df.corr()
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.7, 1.0]})

    # Left panel: raw series
    ax_left = axes[0]
    for i, col in enumerate(var_names):
        ax_left.plot(data_df.index, data_df[col], color=f"C{i}", lw=1.5, label=col)
    ax_left.set_title("Raw Time Series")
    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("Value")
    ax_left.legend(frameon=False, ncol=min(3, len(var_names)), loc="upper right")

    # Right panel: correlation heatmap
    ax_right = axes[1]
    im = ax_right.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax_right.set_xticks(range(len(var_names)))
    ax_right.set_yticks(range(len(var_names)))
    ax_right.set_xticklabels(var_names, rotation=0)
    ax_right.set_yticklabels(var_names)
    ax_right.set_title("Contemporaneous Correlation")
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            ax_right.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_stability_eigenvalues(eigvals: np.ndarray, figsize=(4.4, 4.4)):
    """Plot companion eigenvalues against the unit circle."""
    theta = np.linspace(0, 2 * np.pi, 300)
    unit_x, unit_y = np.cos(theta), np.sin(theta)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(unit_x, unit_y, "k--", lw=1.2, label="Unit circle")
    ax.scatter(eigvals.real, eigvals.imag, c="C3", s=40, label="Eigenvalues")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Companion Eigenvalues")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_irf(
    irf: np.ndarray,
    var_names=None,
    shock_names=None,
    sharey: bool = True,
    figsize=(7, 6),
    t_ref=None,
    layout: str = "matrix",
):
    """
    Plot IRFs as a y-by-u matrix:
    - Row i: response variable y_i
    - Column j: structural shock u_j
    - Cell (i, j): response path of y_i to u_j
    """
    H_plus_1, k, _ = irf.shape
    horizons = np.arange(H_plus_1)

    if var_names is None:
        var_names = [f"y{i + 1}" for i in range(k)]
    if shock_names is None:
        shock_names = [f"u{i + 1}" for i in range(k)]

    if layout == "response_panels":
        fig, axes = plt.subplots(
            k,
            1,
            figsize=figsize,
            sharex=True,
            sharey=sharey,
            squeeze=False,
        )
        for i in range(k):
            ax = axes[i, 0]
            for j in range(k):
                ax.plot(horizons, irf[:, i, j], lw=2.0, color=f"C{j}", label=shock_names[j])
            ax.axhline(0, color="black", lw=0.9, alpha=0.75)
            ax.set_ylabel(f"{var_names[i]} response")
            if i == 0:
                ax.legend(frameon=True, ncol=min(k, 4), loc="upper right", title="Shock (u)")
            if i == k - 1:
                ax.set_xlabel("Horizon")
    else:
        fig, axes = plt.subplots(
            k,
            k,
            figsize=figsize,
            sharex=True,
            sharey=sharey,
            squeeze=False,
        )

        row_lims = {}
        if not sharey:
            for i in range(k):
                vals = irf[:, i, :]
                lim = 1.08 * np.max(np.abs(vals))
                row_lims[i] = (-lim, lim) if lim > 0 else (-1.0, 1.0)

        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                ax.plot(horizons, irf[:, i, j], lw=2.0, color=f"C{i}")
                ax.axhline(0, color="black", lw=0.9, alpha=0.75)
                if not sharey:
                    ax.set_ylim(*row_lims[i])
                if i == 0:
                    ax.set_title(f"Shock: {shock_names[j]}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{var_names[i]} response")
                if i == k - 1:
                    ax.set_xlabel("Horizon")

    if t_ref is None:
        title = "Impulse Responses (y <- u)"
    else:
        title = f"Impulse Responses (y <- u), shock at t={t_ref}"
    fig.suptitle(title, y=1.01, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_irf_horizon_heatmap(
    irf: np.ndarray,
    h_plot: int,
    var_names,
    shock_names,
    figsize=(4.0, 3.0),
):
    """Plot IRF matrix at horizon h (rows: response y_i, columns: shock u_j)."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(irf[h_plot], cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(shock_names)), shock_names, rotation=45, ha="right")
    ax.set_yticks(range(len(var_names)), var_names)
    ax.set_xlabel("Structural shock (u)")
    ax.set_ylabel("Response variable (y)")
    ax.set_title(f"IRF matrix at horizon h={h_plot}")
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            ax.text(j, i, f"{irf[h_plot, i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_irf_by_tref(
    irf: np.ndarray,
    response_idx: int,
    shock_idx: int,
    t_refs,
    var_names=None,
    shock_names=None,
    max_h: int = None,
    figsize=(7.2, 4.0),
):
    """
    Plot one IRF path with multiple reference times:
    - x-axis: calendar time t (not horizon h)
    - hue/color: reference shock time t_ref
    """
    H_plus_1, k, _ = irf.shape
    if max_h is None:
        max_h = H_plus_1 - 1
    if max_h < 0 or max_h >= H_plus_1:
        raise ValueError(f"max_h must be in [0, {H_plus_1 - 1}], got {max_h}")

    if var_names is None:
        var_names = [f"y{i + 1}" for i in range(k)]
    if shock_names is None:
        shock_names = [f"u{i + 1}" for i in range(k)]

    path = irf[: max_h + 1, response_idx, shock_idx]
    horizons = np.arange(max_h + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for n, t_ref in enumerate(t_refs):
        t_axis = t_ref + horizons
        ax.plot(t_axis, path, lw=2.0, color=f"C{n % 10}", label=f"t_ref={t_ref}")

    ax.axhline(0, color="black", lw=0.9, alpha=0.75)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Response")
    ax.set_title(f"{var_names[response_idx]} response to {shock_names[shock_idx]}")
    ax.legend(frameon=True, title="Reference time")
    plt.tight_layout()
    plt.show()
