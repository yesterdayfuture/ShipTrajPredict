'''

测试使用 mqtt 订阅消息
pip install paho-mqtt
'''

import paho.mqtt.client as mqtt
import json, signal, sys

TOPIC = "demo/+/temp"
BROKER, PORT = "broker.emqx.io", 1883

def on_connect(client, userdata, flags, rc, props=None):
    print("CONNACK:", rc)
    client.subscribe(TOPIC, qos=1)          # 支持通配符

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        print(f"{msg.topic} -> {data}")
    except Exception as e:
        print("Bad payload:", e)

def on_disconnect(client, userdata, rc):
    print("Disconnected, rc=", rc)

client = mqtt.Client(clean_session=False, reconnect_on_failure=True, client_id="kkk")
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.enable_logger()

# 优雅退出
signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))

client.connect(BROKER, PORT, 60)
client.loop_forever()   # 阻塞主线程