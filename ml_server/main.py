from typing import Union

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect

from service import SocketService

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "hou27"}


@app.websocket("/count_pushups")
async def websocket_endpoint_count_pushups(
    websocket: WebSocket, socket_service: SocketService = Depends(SocketService)
):
    await socket_service.count_pushups(websocket)


@app.websocket("/count_squat")
async def websocket_endpoint_count_squat(
    websocket: WebSocket, socket_service: SocketService = Depends(SocketService)
):
    await socket_service.count_squat(websocket)
