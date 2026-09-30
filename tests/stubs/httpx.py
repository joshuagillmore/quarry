"""Stub for httpx (the crawler's plain-HTTP fallback fetch). Unit tests never
touch the network: get() refuses unless a test monkeypatches it."""


class HTTPError(Exception):
    pass


class ConnectError(HTTPError):
    pass


def get(url, **kwargs):
    raise ConnectError(f"network disabled in unit tests: {url}")
