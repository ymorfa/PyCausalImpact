import pandas as pd
import matplotlib.pyplot as plt

try:  # SciPy is an optional dependency; fall back gracefully if missing
    from scipy.stats import norm
except Exception:  # pragma: no cover - handled in tests when SciPy absent
    norm = None

class ReportGenerator:
    """
    Handles reporting: summary tables, narratives, and plotting.
    """

    def __init__(self, results: pd.DataFrame, alpha: float = 0.05):
        self.results = results
        self.alpha = alpha

    def generate(self, plot: bool = True):
        """Generate summary table and optional plot."""
        summary_table = self._build_summary_table()
        if plot:
            self._plot_results()
        return summary_table

    def _build_summary_table(self):
        """Compute average, cumulative effects and confidence intervals."""
        df = self.results

        avg_obs = df["observed"].mean()
        cum_obs = df["observed"].sum()

        avg_pred = df["predicted"].mean()
        cum_pred = df["predicted"].sum()

        avg_pred_lower = (
            df["predicted_lower"].mean() if "predicted_lower" in df.columns else pd.NA
        )
        cum_pred_lower = (
            df["predicted_lower"].sum() if "predicted_lower" in df.columns else pd.NA
        )
        avg_pred_upper = (
            df["predicted_upper"].mean() if "predicted_upper" in df.columns else pd.NA
        )
        cum_pred_upper = (
            df["predicted_upper"].sum() if "predicted_upper" in df.columns else pd.NA
        )

        avg_abs = df["point_effect"].mean()
        avg_abs_lower = (
            df["point_effect_lower"].mean() if "point_effect_lower" in df.columns else pd.NA
        )
        avg_abs_upper = (
            df["point_effect_upper"].mean() if "point_effect_upper" in df.columns else pd.NA
        )

        cum_abs = df["cumulative_effect"].iloc[-1]
        cum_abs_lower = (
            df["cumulative_effect_lower"].iloc[-1]
            if "cumulative_effect_lower" in df.columns
            else pd.NA
        )
        cum_abs_upper = (
            df["cumulative_effect_upper"].iloc[-1]
            if "cumulative_effect_upper" in df.columns
            else pd.NA
        )

        avg_rel = avg_abs / avg_pred if avg_pred != 0 else pd.NA
        cum_rel = cum_abs / cum_pred if cum_pred != 0 else pd.NA

        avg_rel_lower = (
            avg_abs_lower / avg_pred_upper
            if (
                "point_effect_lower" in df.columns
                and "predicted_upper" in df.columns
                and not pd.isna(avg_pred_upper)
                and avg_pred_upper != 0
            )
            else pd.NA
        )
        avg_rel_upper = (
            avg_abs_upper / avg_pred_lower
            if (
                "point_effect_upper" in df.columns
                and "predicted_lower" in df.columns
                and not pd.isna(avg_pred_lower)
                and avg_pred_lower != 0
            )
            else pd.NA
        )

        cum_rel_lower = (
            cum_abs_lower / cum_pred_upper
            if (
                "cumulative_effect_lower" in df.columns
                and "predicted_upper" in df.columns
                and not pd.isna(cum_pred_upper)
                and cum_pred_upper != 0
            )
            else pd.NA
        )
        cum_rel_upper = (
            cum_abs_upper / cum_pred_lower
            if (
                "cumulative_effect_upper" in df.columns
                and "predicted_lower" in df.columns
                and not pd.isna(cum_pred_lower)
                and cum_pred_lower != 0
            )
            else pd.NA
        )

        summary = pd.DataFrame(
            {
                "observed": [avg_obs, cum_obs],
                "predicted": [avg_pred, cum_pred],
                "predicted_lower": [avg_pred_lower, cum_pred_lower],
                "predicted_upper": [avg_pred_upper, cum_pred_upper],
                "abs_effect": [avg_abs, cum_abs],
                "abs_effect_lower": [avg_abs_lower, cum_abs_lower],
                "abs_effect_upper": [avg_abs_upper, cum_abs_upper],
                "rel_effect": [avg_rel, cum_rel],
                "rel_effect_lower": [avg_rel_lower, cum_rel_lower],
                "rel_effect_upper": [avg_rel_upper, cum_rel_upper],
            },
            index=["average", "cumulative"],
        )

        # --- Significance metrics -------------------------------------------------
        p_values = [pd.NA, pd.NA]
        causal_probs = [pd.NA, pd.NA]

        if norm is not None:
            # z critical value for two-tailed interval
            z_crit = norm.ppf(1 - self.alpha / 2)
            effects = [
                (avg_abs, avg_abs_lower, avg_abs_upper, 0),
                (cum_abs, cum_abs_lower, cum_abs_upper, 1),
            ]
            for mean, lower, upper, idx in effects:
                if pd.isna(lower) or pd.isna(upper) or pd.isna(mean):
                    continue
                denom = (upper - lower) / (2 * z_crit) if z_crit else None
                if not denom or denom <= 0:
                    continue
                z_score = mean / denom
                p = 2 * norm.sf(abs(z_score))
                p_values[idx] = p
                causal_probs[idx] = 1 - p / 2  # probability effect has observed sign

        summary["p_value"] = p_values
        summary["causal_probability"] = causal_probs

        return summary

    def _plot_results(self):
        """3-panel plot: observed vs predicted, pointwise effect, cumulative effect."""
        df = self.results
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax = axes[0]
        ax.plot(df.index, df["observed"], label="observed")
        ax.plot(df.index, df["predicted"], label="predicted")
        if "predicted_lower" in df.columns and "predicted_upper" in df.columns:
            ax.fill_between(
                df.index,
                df["predicted_lower"],
                df["predicted_upper"],
                color="gray",
                alpha=0.3,
            )
        ax.set_title("Observed vs Predicted")
        ax.legend()

        ax = axes[1]
        ax.plot(df.index, df["point_effect"], label="pointwise effect")
        if "point_effect_lower" in df.columns and "point_effect_upper" in df.columns:
            ax.fill_between(
                df.index,
                df["point_effect_lower"],
                df["point_effect_upper"],
                color="gray",
                alpha=0.3,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Pointwise Effect")

        ax = axes[2]
        ax.plot(df.index, df["cumulative_effect"], label="cumulative effect")
        if (
            "cumulative_effect_lower" in df.columns
            and "cumulative_effect_upper" in df.columns
        ):
            ax.fill_between(
                df.index,
                df["cumulative_effect_lower"],
                df["cumulative_effect_upper"],
                color="gray",
                alpha=0.3,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Cumulative Effect")

        plt.tight_layout()
        return fig
