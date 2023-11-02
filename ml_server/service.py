from fastapi import WebSocket, WebSocketDisconnect

from connection_manager import ConnectionManager
from squatCountMachine import SquatCountMachine
from countMachine import CountMachine


class SocketService:
    def __init__(self):
        self.manager = ConnectionManager()

    async def count_pushups(self, websocket: WebSocket):
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await self.manager.send_personal_message(f"Received:{data}", websocket)
        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            await self.manager.send_personal_message("Bye!!!", websocket)

    async def count_squat(self, websocket: WebSocket):
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await self.manager.send_personal_message(f"Received:{data}", websocket)
        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            await self.manager.send_personal_message("Bye!!!", websocket)
