"""
axes-client: Python client for the Axes query API.

    from axes import sql, Client
"""

from axes.client import Client
from axes.sql import SqlResult, sql

__all__ = ["Client", "sql", "SqlResult"]
