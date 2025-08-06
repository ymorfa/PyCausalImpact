import pandas as pd
import matplotlib.pyplot as plt

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
        # TODO: Aggregate observed, predicted, effect, and CIs
        pass

    def _plot_results(self):
        """3-panel plot: observed vs predicted, pointwise effect, cumulative effect."""
        # TODO: Implement matplotlib plotting
        pass
