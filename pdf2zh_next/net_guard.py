"""HTTP sessions that may only connect to an allowed class of IP addresses.

Used to keep outbound connections within policy: user-supplied download links
may only reach the public internet, and the term extraction Ollama server may
only be on the local network.
"""

import ipaddress
import socket
from collections.abc import Callable

import requests
import urllib3


class AddressNotAllowedError(OSError):
    """The host resolved to, or connected to, an address outside the policy."""


def _ip(ip_str: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    ip = ipaddress.ip_address(ip_str.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip


def is_public_ip(ip_str: str) -> bool:
    ip = _ip(ip_str)
    return ip.is_global and not ip.is_multicast


def is_local_network_ip(ip_str: str) -> bool:
    """Private, loopback, link-local or CGNAT (e.g. Tailscale) address."""
    ip = _ip(ip_str)
    return not ip.is_global and not ip.is_multicast and not ip.is_unspecified


def guarded_session(
    address_allowed: Callable[[str], bool], reject_own_address: bool = False
) -> requests.Session:
    """Return a session whose connections only go to allowed addresses.

    The host urllib3 will connect to is resolved and checked before connecting
    (so disallowed hosts are never contacted), and the connected socket is
    checked again before any request bytes are sent (so DNS rebinding cannot
    slip through either). Redirects open new connections and go through the
    same checks. Proxies are disabled: through a proxy the connected peer would
    be the proxy, and the target address could not be checked.

    reject_own_address also refuses connections to this host itself on one of
    its own addresses (e.g. a global IPv6 address that passes is_public_ip).
    Refused connections raise AddressNotAllowedError, which requests reports as
    a ConnectionError.
    """

    class GuardedConnectionMixin:
        def _new_conn(self):
            infos = socket.getaddrinfo(
                self._dns_host, self.port, proto=socket.IPPROTO_TCP
            )
            addresses = [info[4][0] for info in infos]
            if not addresses or not all(address_allowed(a) for a in addresses):
                raise AddressNotAllowedError(f"Address not allowed: {self.host}")
            # Connect to the address checked above instead of resolving the
            # name again, so DNS rebinding cannot change the target. TLS SNI
            # and certificate checks still use the original host name.
            sock = urllib3.util.connection.create_connection(
                (addresses[0], self.port),
                self.timeout,
                source_address=self.source_address,
                socket_options=self.socket_options,
            )
            peer, local = sock.getpeername()[0], sock.getsockname()[0]
            if not address_allowed(peer) or (reject_own_address and peer == local):
                sock.close()
                raise AddressNotAllowedError(f"Address not allowed: {self.host}")
            return sock

    class HTTPConnection(GuardedConnectionMixin, urllib3.connection.HTTPConnection):
        pass

    class HTTPSConnection(GuardedConnectionMixin, urllib3.connection.HTTPSConnection):
        pass

    class HTTPConnectionPool(urllib3.HTTPConnectionPool):
        ConnectionCls = HTTPConnection

    class HTTPSConnectionPool(urllib3.HTTPSConnectionPool):
        ConnectionCls = HTTPSConnection

    class Adapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            self.poolmanager.pool_classes_by_scheme = {
                "http": HTTPConnectionPool,
                "https": HTTPSConnectionPool,
            }

    session = requests.Session()
    session.trust_env = False
    adapter = Adapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
