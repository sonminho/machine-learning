import websocket
from threading import Thread
import time
import json

def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        for i in range(3):
            time.sleep(1)
            msg = {'msgType' : 'POSITION_SUB', 'target' : '1', 'lat' : '12.4', 'lng' : '37.2'}
            json_msg = json.dumps(msg)
            ws.send("{}".format(json_msg))
        time.sleep(1)
        #ws.close()
    print("thread terminating...")
    
    Thread(target=run).start()

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:8080/monitor/car"
        ,on_open = on_open,
        on_message = on_message,
        on_error = on_error,
        on_close = on_close)
    ws.run_forever()
    ws.close()
    print("종료합니다.")