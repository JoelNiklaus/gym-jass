import asyncio
import logging
import signal
import threading
import time
from multiprocessing import Process

import websockets
from threading import Thread


logger = logging.getLogger(__name__)

class TournamentServer:

    def __init__(self, tournament):
        self.tournament = tournament
        self.hostname = "localhost"
        self.port = 9000
        self.close = threading.Event()
        # start the server in a new thread
        #self.event_loop = asyncio.new_event_loop()
        #thread = Thread(target=self.start, args=(self.event_loop,))
        #thread = Thread(target=self.start)

    async def start(self, stop):
        print("started server")
        async with websockets.serve(self.wait_for_client, self.hostname, self.port, close_timeout=1):
            await stop

    async def wait_for_client(self, websocket, path):
        message = await websocket.recv()
        if message == "reset":
            if self.tournament is not None:
                self.reset()
        if message == "reward":
            if self.tournament is not None:
                reward = self.tournament.teams[0].points - self.tournament.teams[1].points
                await websocket.send(reward)

    def stop_server(self):
        self.stop.set_result(None)
        logger.error("server dfsqdf")
        self.shutdown()
        for task in asyncio.Task.all_tasks():
            print(task)
        self.event_loop.stop()
        self.event_loop.close()

        #asyncio.get_event_loop().close()
        #asyncio.get_event_loop().call_soon_threadsafe(asyncio.get_event_loop().stop())

    def shutdown(self):
        print('received stop signal, cancelling tasks...')
        for task in asyncio.Task.all_tasks():
            print(task)
            task.cancel()
        print('bye, exiting in a minute...')

    def reset(self):
        if len(self.tournament.teams) == 0:
            thread = Thread(target=self.tournament.play)
            thread.start()
            print("tournament started")

        self.tournament.teams[0].points = 0
        self.tournament.teams[1].points = 0
