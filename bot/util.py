import pandas as pd
from PIL import Image
import os

import subprocess
import shlex
from typing import Optional
import time
import tqdm
import platform


def save_images_from_dataframe(
    df: pd.DataFrame,
    output_dir: str,
    width: int = 1280,
    height: int = 720,
    prefix: str = 'image_'
) -> None:
    """
    Saves images from a DataFrame's 'screen' column (containing image bytes)
    to PNG files with alphabetically sortable filenames.

    Args:
        df (pd.DataFrame): DataFrame with a 'screen' column (bytes)
        output_dir (str): Directory to save the PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Saving images"):
        with open(row['screen'], 'rb') as f:
            image_bytes = f.read()

        base_filename = f"image_{index:05d}"

        filename = f"{base_filename}.png"
        filepath = os.path.join(output_dir, filename)

        try:
            image = Image.frombuffer('RGBA', (width, height), image_bytes, 'raw', 'RGBA', 0, 1)
            image.save(filepath, format='PNG')

        except Exception as e:
            print(f"Error processing row {index}: {e}")


def start_server_process(
    auth_port: int,
    game_port: int,
    players: int,
    max_game_seconds: int,
    seed: int,
    low_gpu: bool,
    headless: bool,
    server_executable: str = None
) -> Optional[subprocess.Popen]:
    """
    Starts the server executable as a background process with specified arguments.

    Args:
        auth_port: The port number for authentication.
        game_port: The port number for the game.
        players: The number of players required.
        max_game_seconds: The maximum duration of a game in seconds.
        server_executable: The path to the server executable file. If None, will be determined based on OS.

    Returns:
        A subprocess.Popen object representing the started process,
        or None if the process could not be started.
    """
    if server_executable is None:
        # Determine the correct executable based on OS
        if platform.system() == "Windows":
            server_executable = "../server.exe"
        else:
            server_executable = "../server"

    command = [
        server_executable,
        "--auth-port", str(auth_port),
        "--game-port", str(game_port),
        "--players", str(players),
        "--seed", str(seed),
        "--max-game-seconds", str(max_game_seconds),
    ]
    if low_gpu:
        command.append("--low-gpu")

    if headless:
        command.append("--headless")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        print(f"Started server process with PID: {process.pid}")
        print(f"Command: {' '.join(shlex.quote(arg) for arg in command)}")
        return process
    except FileNotFoundError:
        print(f"Error: Server executable not found at '{server_executable}'")
        return None
    except Exception as e:
        print(f"Error starting server process: {e}")
        return None


def start_client_process(
    auth_port: int,
    server_host: str,
    client_port: int,
    player_name: str,
    grpc_port: int,
    seed: int,
    low_gpu: bool,
    executable: str = None
) -> Optional[subprocess.Popen]:
    """
    Starts the client application as a separate process.

    Args:
        auth_port: The port number for the authentication service.
        server_host: The hostname or IP address of the game server.
        client_port: The port number the client will use.
        player_name: The desired name for the player in the game.
        grpc_port: The port number for the gRPC server.
        seed: The seed for the game world.
        low_gpu: Whether to use low GPU mode.
        executable: The file path to the client executable. If None, will be determined based on OS.

    Returns:
        An optional `subprocess.Popen` object representing the started client
        process. Returns `None` if the executable cannot be found at the specified
        path or if any other error occurs during process creation.
    """
    if executable is None:
        # Determine the correct executable based on OS
        if platform.system() == "Windows":
            executable = "../client.exe"
        else:
            executable = "../client"

    command = [
        executable,
        "--auth-port", str(auth_port),
        "--server", server_host,
        "--client-port", str(client_port),
        "--grpc-port", str(grpc_port),
        "--name", player_name,
        "--seed", str(seed),
    ]

    if low_gpu:
        command.append("--low-gpu")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        print(f"Started client process with PID: {process.pid}")
        print(f"Command: {' '.join(shlex.quote(arg) for arg in command)}")
        return process
    except FileNotFoundError:
        print(f"Error: Client executable not found at '{executable}'")
        return None
    except Exception as e:
        print(f"Error starting client process: {e}")
        return None
