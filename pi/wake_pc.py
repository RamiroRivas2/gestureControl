"""
Wake-on-LAN utility.

Sends a magic packet to the PC over the direct Ethernet link (eth0).
Used when the PC is off and the WebSocket agent isn't running.
"""

import json
import os
from wakeonlan import send_magic_packet


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def wake_pc(mac_address=None):
    """Send WoL magic packet over the direct Ethernet link."""
    if mac_address is None:
        config = load_config()
        mac_address = config["pc_mac"]

    print(f"Sending Wake-on-LAN packet to {mac_address}...")
    # Send on the direct Ethernet subnet (10.0.0.x)
    send_magic_packet(mac_address, ip_address="10.0.0.255")
    print("Magic packet sent! PC should boot in a few seconds.")


if __name__ == "__main__":
    wake_pc()
