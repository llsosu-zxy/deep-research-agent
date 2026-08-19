from __future__ import annotations

import asyncio
import json

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8000/ws/research"
    async with websockets.connect(uri) as websocket:
        await websocket.send(
            json.dumps({"question": "What does Grab AI interns work on?"})
        )
        while True:
            message = json.loads(await websocket.recv())
            print(message.get("event"))
            if message.get("event") == "complete":
                print("sources:", len(message["state"]["context"]))
                break


if __name__ == "__main__":
    asyncio.run(main())
