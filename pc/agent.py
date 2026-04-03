"""
PC-side WebSocket agent.

Listens for gesture commands from the Pi and executes corresponding actions.
Run this on your gaming PC before starting gesture_control.py on the Pi.
"""

import asyncio
import json
import websockets
from actions import execute_action

HOST = "0.0.0.0"
PORT = 8765


async def handle_client(websocket):
    """Handle incoming gesture commands from a Pi client."""
    addr = websocket.remote_address
    print(f"Pi connected from {addr[0]}:{addr[1]}")

    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
                print(f"\nReceived: {payload.get('label', 'unknown')}")
                execute_action(payload)
            except json.JSONDecodeError:
                print(f"Invalid message: {message}")
            except Exception as e:
                print(f"Action error: {e}")
    except websockets.exceptions.ConnectionClosed:
        print(f"Pi disconnected ({addr[0]})")


async def main():
    print(f"=== PC Gesture Agent ===")
    print(f"Listening on ws://{HOST}:{PORT}")
    print("Waiting for Pi connection...\n")

    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
