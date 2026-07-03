"""
axes-client: Python client for the Axes query API.

    from axes import sql, Client

The ``axes.agent`` framework ships in this same distribution as a subpackage.
"""

from axes.client import Client
from axes.sql import SqlResult, sql

__all__ = ["Client", "sql", "SqlResult"]
