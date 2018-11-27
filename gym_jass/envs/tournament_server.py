import asyncio
import logging

import websockets
from threading import Thread


logger = logging.getLogger(__name__)


class TournamentServer:

    def __init__(self, tournament):
        self.tournament = tournament
        self.hostname = "localhost"
        self.port = 9000
        # start the server in a new thread
        new_loop = asyncio.new_event_loop()
        thread = Thread(target=self.start_server, args=(new_loop,))
        thread.start()

    def start_server(self, event_loop):
        asyncio.set_event_loop(event_loop)
        start_server = websockets.serve(self.wait_for_client, self.hostname, self.port)
        event_loop.run_until_complete(start_server)
        event_loop.run_forever()

    async def wait_for_client(self, websocket, path):
        message = await websocket.recv()
        if message == "reset":
            if self.tournament is not None:
                self.reset()
        if message == "reward":
            if self.tournament is not None:
                reward = self.tournament.teams[0].points - self.tournament.teams[1].points
                await websocket.send(reward)

    def reset(self):
        if len(self.tournament.teams) == 0:
            thread = Thread(target=self.tournament.play)
            thread.start()

        self.tournament.teams[0].points = 0
        self.tournament.teams[1].points = 0
