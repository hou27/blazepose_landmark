from fastapi import WebSocket
import json
import time

from squatCountMachine import SquatCountMachine
from pushupCountMachine import PushupCountMachine


class SocketService:
    async def count_pushups(self, websocket: WebSocket):
        pushup_count_machine = PushupCountMachine()
        await websocket.accept()

        last_time = time.time()

        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            response = pushup_count_machine.count(data)
            print(response)
            curr_time = time.time()
            if curr_time - last_time >= 0.5:
                last_time = curr_time
                await websocket.send_json(response)

    async def count_squat(self, websocket: WebSocket):
        squat_count_machine = SquatCountMachine()
        await websocket.accept()

        last_time = time.time()

        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            response = squat_count_machine.count(data)
            print(response)
            curr_time = time.time()
            if curr_time - last_time >= 0.5:
                last_time = curr_time
                await websocket.send_json(response)
