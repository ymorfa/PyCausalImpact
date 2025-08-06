import pandas as pd
 
def validate_periods(data: pd.DataFrame, pre_period, post_period):
    """Validate that ``pre_period`` and ``post_period`` lie within ``data``.
    
    Parameters
    ----------
    data: pd.DataFrame
        Source data with an index representing the time axis.
    pre_period: tuple
        Two element tuple defining the start and end of the pre-intervention
        window.
    post_period: tuple
        Two element tuple defining the start and end of the post-intervention
        window.
    
    Raises
    ------
    ValueError
        If period boundaries fall outside of ``data`` or if the periods
        overlap. 
    """    

    index = data.index

    def _loc(value):
        try:
            return index.get_loc(value)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("Period boundaries must be within the data index") from exc

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    pre_start_loc = _loc(pre_start)
    pre_end_loc = _loc(pre_end)
    post_start_loc = _loc(post_start)
    post_end_loc = _loc(post_end)

    if pre_start_loc > pre_end_loc or post_start_loc > post_end_loc:
        raise ValueError("Period start must not be after period end")

    if pre_end_loc >= post_start_loc:
        raise ValueError("Pre and post periods must be non-overlapping and ordered")


def split_pre_post(data: pd.DataFrame, pre_period, post_period):
    """Return pre- and post-intervention slices of ``data`` using label indices."""
    pre_start, pre_end = pre_period
    post_start, post_end = post_period
    pre_data = data.loc[pre_start:pre_end]
    post_data = data.loc[post_start:post_end]
    return pre_data, post_data