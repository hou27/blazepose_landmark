from fastapi import WebSocket
import json

from squatCountMachine import SquatCountMachine
from pushupCountMachine import PushupCountMachine


class SocketService:
    async def count_pushups(self, websocket: WebSocket):
        pushup_count_machine = PushupCountMachine()
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            curr_cnt = pushup_count_machine.count(data)
            print(curr_cnt)
            await websocket.send_text(str(curr_cnt))

    async def count_squat(self, websocket: WebSocket):
        squat_count_machine = SquatCountMachine()
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            curr_cnt: int = squat_count_machine.count(data)
            await websocket.send_text(str(curr_cnt))
