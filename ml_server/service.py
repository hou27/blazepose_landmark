from fastapi import WebSocket

from squatCountMachine import SquatCountMachine
from countMachine import CountMachine


class SocketService:
    async def count_pushups(self, websocket: WebSocket):
        # await self.manager.connect(websocket)
        pushup_count_machine = CountMachine()
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            print(data)
            curr_cnt: int = pushup_count_machine.count(data)
            await websocket.send_text(curr_cnt)

    async def count_squat(self, websocket: WebSocket):
        # await self.manager.connect(websocket)
        squat_count_machine = SquatCountMachine()
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            curr_cnt: int = squat_count_machine.count(data)
            await websocket.send_text(curr_cnt)
