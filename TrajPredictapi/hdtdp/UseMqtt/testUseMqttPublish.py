'''

测试使用 mqtt 发送消息
pip install paho-mqtt
'''


import paho.mqtt.client as mqtt
import time

TOPIC = "demo/room1/temp"
BROKER, PORT = "broker.emqx.io", 1883

def on_connect(client, userdata, flags, rc, props=None):
    print("Connected" if rc == 0 else f"Connect failed {rc}")

client = mqtt.Client()
client.on_connect = on_connect
client.connect(BROKER, PORT, 60)   # keepalive=60s
client.loop_start()                # 后台线程处理网络

for t in range(10):
    payload = f'{{"temp":{20+t}}}'
    msg = client.publish(TOPIC, payload, qos=1)
    msg.wait_for_publish()         # 等待完成
    time.sleep(1)

client.loop_stop()
client.disconnect()







