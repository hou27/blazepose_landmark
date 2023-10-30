from typing import Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from connection_manager import ConnectionManager
from squartCountMachine import SquartCountMachine
from countMachine import CountMachine

app = FastAPI()

manager = ConnectionManager()
pushupsCountMachine = CountMachine()
squartCountMachine = SquartCountMachine()


@app.get("/")
def read_root():
    return {"Hello": "hou27"}


@app.websocket("/count_pushups")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Received:{data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_message("Bye!!!", websocket)


@app.websocket("/count_squat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Received:{data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_message("Bye!!!", websocket)
