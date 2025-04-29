import logging
import util
import marble_client
import click
import subprocess
from typing import Optional
from concurrent.futures import ProcessPoolExecutor


# Import the generated modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')


class SafeFormatter(logging.Formatter):
    def format(self, record):
        record.exc_type = getattr(record, 'exc_type', '')
        record.exc_msg = getattr(record, 'exc_msg', '')
        return super().format(record)


# Configure logging using the custom formatter
handler = logging.StreamHandler()

logger = logging.getLogger()
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@click.command()
@click.option('--no-server', default=False, is_flag=True, help='Do not start the server')
@click.option('--clients', default=1, help='Number of clients to start')
@click.option('--game-seconds', default=180, help='Time the game runs until a winner is declared')
@click.option('--seed', default=1234, help='Seed for the game world generation')
@click.option('--server-headless', default=False, is_flag=True, help='Run the server in headless mode')
@click.option('--low-gpu', default=False, is_flag=True, help='Run in low GPU mode')
@click.option('--server-ip', default='127.0.0.1', help='IP address of the server')
@click.option('--model-path', default='marble_cnn.pth', help='Path to the trained CNN model file')
def run(no_server: bool, clients: int, game_seconds: int, seed: int, server_headless: bool, low_gpu: bool, server_ip: str, model_path: str):
    if not no_server:
        server = util.start_server_process(4000, 5000, clients, game_seconds, seed, low_gpu, server_headless)

    with ProcessPoolExecutor(max_workers=clients) as executor:
        list(executor.map(run_client, [(i, seed, low_gpu, server_ip, model_path) for i in range(clients)]))
    if server:
        server.kill()


def run_client(args: (int, int, bool, str, str)) -> Optional[subprocess.Popen]:
    client_id, seed, low_gpu, server_ip, model_path = args
    name = 'A' + str(client_id)
    client = util.start_client_process(4000, server_ip, 5001 + client_id, name, 50051 + client_id, seed, low_gpu)

    bot = marble_client.MarbleClient("localhost", str(50051 + client_id), 'raw_screens_' + str(client_id), name, model_path)
    try:
        bot.run_interaction_loop()
    finally:
        df = bot.get_records_as_dataframe()
        df.to_parquet(f'marble_client_records_{client_id}.parquet', index=False)
        util.save_images_from_dataframe(df, f'output_images_{client_id}')

    if client:
        client.kill()
        logger.info(f'Client {client.pid} killed')
    else:
        logger.error('Client process failed to start or was None')


if __name__ == '__main__':
    run()
