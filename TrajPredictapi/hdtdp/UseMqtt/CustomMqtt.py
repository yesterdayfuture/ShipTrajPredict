#!/usr/bin/env python3
'''

当前文件不可用🙅，待后续完善处理

'''
import asyncio, struct, logging, signal, sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ---------- 工具 ----------
def remain_decode(buf):
    """读 MQTT 剩余长度"""
    mult, val, idx = 1, 0, 0
    while True:
        b = buf[idx]
        idx += 1
        val += (b & 0x7F) * mult
        if (b & 0x80) == 0:
            return val, idx
        mult *= 128

def remain_encode(x):
    """写 MQTT 剩余长度"""
    out = bytearray()
    while True:
        b = x % 128
        x //= 128
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return out

# ---------- 服务器 ----------
class MQTTServer:
    def __init__(self):
        self.clients = {}               # client_id -> Client
        self.subs = defaultdict(list)   # topic -> [Client, ...]

    async def client_handler(self, reader, writer):
        peer = writer.get_extra_info('peername')
        logging.info('[%s] connected', peer)

        data = await reader.read(1024)
        logging.info("data = [%s] ", data.decode("utf-8"))

        client = Client(reader, writer, self)
        self.clients[client.cid] = client
        try:
            await client.loop()
        except asyncio.IncompleteReadError:
            pass
        finally:
            self.disconnect_client(client)
            writer.close()
            await writer.wait_closed()
            logging.info('[%s] disconnected', peer)

    def disconnect_client(self, client):
        self.clients.pop(client.cid, None)
        # 清理订阅
        for topic, clients in self.subs.items():
            try:
                clients.remove(client)
            except ValueError:
                pass

    def publish_to_subscribers(self, topic, payload, qos, retain):
        # 简易遍历：无通配符，无 retain 存储
        matched = [c for t, cs in self.subs.items() if t == topic for c in cs]
        for c in matched:
            c.publish(topic, payload, qos)

# ---------- 单客户端状态 ----------
class Client:
    def __init__(self, reader, writer, server):
        self.reader = reader
        self.writer = writer
        self.server = server
        self.cid = None
        self.keepalive = 60
        self.last_pkt = asyncio.get_event_loop().time()

    async def loop(self):
        while True:
            # 1. 先读 1 字节固定头
            hdr = await self.reader.readexactly(1)
            pkt_type = (hdr[0] >> 4) & 0x0F

            # 2. 继续读 1~4 字节“剩余长度”
            remain, offset = remain_decode(await self.reader.read(4))
            # 3. 一次性读完整 Payload
            payload = await self.reader.readexactly(remain)
            self.last_pkt = asyncio.get_event_loop().time()
            await self.dispatch(pkt_type, payload)

    async def dispatch(self, typ, payload):
        if typ == 1:           # CONNECT
            await self.handle_connect(payload)
        elif typ == 3:         # PUBLISH
            await self.handle_publish(payload)
        elif typ == 8:         # SUBSCRIBE
            await self.handle_subscribe(payload)
        elif typ == 12:        # PINGREQ
            self.writer.write(b'\xD0\x00')  # PINGRESP
            await self.writer.drain()
        elif typ == 14:        # DISCONNECT
            raise asyncio.IncompleteReadError(None, None)

    async def handle_connect(self, p):
        # 极简解析：跳过可变头，直接读 Payload
        i = 0
        proto_len = struct.unpack('>H', p[i:i+2])[0]; i+=2
        proto_name = p[i:i+proto_len]; i+=proto_len
        proto_level = p[i]; i+=1
        connect_flags = p[i]; i+=1
        keepalive   = struct.unpack('>H', p[i:i+2])[0]; i+=2
        self.keepalive = keepalive
        # Payload
        cid_len = struct.unpack('>H', p[i:i+2])[0]; i+=2
        self.cid = p[i:i+cid_len].decode(); i+=cid_len
        # 回 CONNACK
        self.writer.write(b'\x20\x02\x00\x00')
        await self.writer.drain()

    async def handle_publish(self, p):
        i = 0
        topic_len = struct.unpack('>H', p[i:i+2])[0]; i+=2
        topic = p[i:i+topic_len].decode(); i+=topic_len
        # 无报文标识（QoS0）
        payload = p[i:]
        # 直接转发
        self.server.publish_to_subscribers(topic, payload, 0, False)

    async def handle_subscribe(self, p):
        # 同样只支持 QoS0
        i = 0
        pkt_id = struct.unpack('>H', p[i:i+2])[0]; i+=2
        topics = []
        while i < len(p):
            tlen = struct.unpack('>H', p[i:i+2])[0]; i+=2
            tf = p[i:i+tlen].decode(); i+=tlen
            qos = p[i]; i+=1
            topics.append((tf, qos))
            self.server.subs[tf].append(self)
        # SUBACK
        ack = struct.pack('>BH%ds' % len(topics), 0x90, 2+len(topics), pkt_id) + bytes(qos for _,qos in topics)
        self.writer.write(ack)
        await self.writer.drain()

    def publish(self, topic, payload, qos):
        # 只实现 QoS0
        hdr = (0x30).to_bytes(1, 'big')           # PUBLISH QoS0
        var = struct.pack('>H', len(topic)) + topic.encode()
        remain = len(var) + len(payload)
        self.writer.write(hdr + remain_encode(remain) + var + payload)
        # 不等待 drain，简易发送

# ---------- 启动 ----------
async def main(host='0.0.0.0', port=1883):
    srv = MQTTServer()
    server = await asyncio.start_server(srv.client_handler, host, port)
    logging.info('MQTT nano-broker listening on %s:%d', host, port)
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logging.info("主动退出本服务")