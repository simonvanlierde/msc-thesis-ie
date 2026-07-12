"""Shared HTTP session with retries for the PDOK/EP acquisition scripts.

PDOK's OGC APIs occasionally return transient 5xx errors mid-pagination, so all
requests go through a session that retries on those with exponential backoff.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def retrying_session(total: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """Return a requests session that retries transient 5xx/connection errors on GET."""
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
