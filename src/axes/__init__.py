"""
axes-client: Python client for the Axes query API.

    from axes import sql, Client

The ``axes.agent`` framework ships in this same distribution as a subpackage.
"""

from axes.client import Client
from axes.sql import SqlResult, sql
from axes.write import AppendResult, append_table_data

__all__ = [
    "AppendResult",
    "Client",
    "SqlResult",
    "append_table_data",
    "sql",
]
