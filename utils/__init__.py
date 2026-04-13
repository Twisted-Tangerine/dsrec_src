"""
Utils Package

This package provides utility functions for the DSRec project, organized into:
- metrics: Evaluation metrics
- helpers: Helper functions (data processing, I/O, model utilities)
- logger: Logging utilities
- earlystop: Early stopping mechanisms
"""

# Import commonly used functions for convenience
from .metrics import (
    metric_report,
    metric_len_report,
    metric_pop_report,
    metric_len_5group,
    metric_pop_5group,
    seq_acc
)

from .helpers import (
    # Data processing
    unzip_data,
    unzip_data_with_user,
    concat_data,
    concat_aug_data,
    concat_data_with_user,
    filter_data,
    random_neq,
    random_neq2,
    # I/O operations
    set_seed,
    record_csv,
    # Model utilities
    get_n_params,
    load_pretrained_model
)

from .logger import Logger
from .earlystop import EarlyStoppingNew

__all__ = [
    # Metrics
    'metric_report',
    'metric_len_report',
    'metric_pop_report',
    'metric_len_5group',
    'metric_pop_5group',
    'seq_acc',
    # Helpers - Data processing
    'unzip_data',
    'unzip_data_with_user',
    'concat_data',
    'concat_aug_data',
    'concat_data_with_user',
    'filter_data',
    'random_neq',
    'random_neq2',
    # Helpers - I/O operations
    'set_seed',
    'record_csv',
    # Helpers - Model utilities
    'get_n_params',
    'load_pretrained_model',
    # Other modules
    'Logger',
    'EarlyStoppingNew',
]
