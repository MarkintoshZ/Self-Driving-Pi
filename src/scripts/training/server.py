import socket
from multiprocessing import Process
import time
from os import system as sys


client_executor = None


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def on_new_connection(client_executor, addr, q):

    print('Accept new connection from %s:%s...' % addr)

    # client_executor.send(bytes('Welcome'.encode('utf-8')))

    while True:
        msg = client_executor.recv(1024).decode('utf-8')
        # test if end connection
        if msg == 'exit':
            print('%s:%s request close' % addr)
            break

        if msg == 'gameover':
            q.put('gameover')

        if msg == 'start':
            q.put('start')

        if is_float(msg):
            q.put(float(msg))

        # print('%s:%s: %s' % (addr[0], addr[1], msg))

    client_executor.close()
    print('Connection from %s:%s closed.' % addr)
    q.put('exit')


def send_msg(msg):
    client_executor.send(bytes(msg.encode('utf-8')))


def start(q):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 9999))
    listener.listen(5)
    print('Waiting for connect...')

    sys("open ../../sim.app/")

    global client_executor
    client_executor, addr = listener.accept()
    t = Process(target=on_new_connection, args=(client_executor, addr, q))
    t.start()
