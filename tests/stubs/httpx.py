"""Stub for httpx (the crawler's plain-HTTP fallback fetch). Unit tests never
touch the network: stream() refuses unless a test monkeypatches it."""


class HTTPError(Exception):
    pass


class ConnectError(HTTPError):
    pass


def stream(method, url, **kwargs):
    raise ConnectError(f"network disabled in unit tests: {method} {url}")
