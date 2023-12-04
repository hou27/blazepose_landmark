from fastapi import Depends, FastAPI, WebSocket

from service import SocketService

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "hou27"}


@app.websocket("/testws")
async def test_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    test_flag = 0
    while True:
        data = await websocket.receive_text()
        print(data)
        await websocket.send_text(str(test_flag))
        test_flag += 1


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
