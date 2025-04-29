import grpc
import pandas as pd
import logging
import time
import os
import uuid
import platform

# Only set QT_QPA_PLATFORM on Linux systems
if platform.system() != "Windows":
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force XCB platform for Linux

import cv2
import numpy as np
# Note: You need to generate the Python protobuf files from your .proto file first.
# Run the following command in your terminal in the directory containing marble.proto:
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. marble.proto
# This will create marble_pb2.py and marble_pb2_grpc.py.
try:
    from proto import service_pb2
    from proto import service_pb2_grpc
except ImportError:
    logging.error("Failed to import generated gRPC modules. "
                  "Did you run 'uv run python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. proto/service.proto'?")
    exit(1)


class MarbleClient:
    """
    A gRPC client for interacting with the MarbleService.

    Connects to a MarbleService instance, allows getting state, sending input,
    running an interaction loop, and storing the state/input history
    in a pandas DataFrame.
    """

    def __init__(self, host: str, port: int, screen_dir: str, name: str):
        """
        Initializes the MarbleClient.

        Args:
            host: The hostname or IP address of the gRPC server.
            port: The port number of the gRPC server.
        """
        self.host = host
        self.port = port
        self.name = name
        # Create an insecure channel to connect to the server
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        # Create a stub (client) for the MarbleService
        self.stub = service_pb2_grpc.MarbleServiceStub(self.channel)
        # List to store (state, input) tuples recorded during the loop
        self.records = []
        self.screen_dir = screen_dir
        os.makedirs(self.screen_dir, exist_ok=True)  # Ensure the directory exists
        print(f"MarbleClient initialized for {self.host}:{self.port}")

    def get_state(self) -> service_pb2.StateResponse:
        """
        Calls the GetState RPC method to retrieve the current state from the server.

        Returns:
            A StateResponse protobuf message.
        """
        try:
            request = service_pb2.GetStateRequest()
            response = self.stub.GetState(request)
            return response
        except grpc.RpcError as e:
            print(f"Error calling GetState: {e}")
            return None  # Or raise the exception

    def send_input(self, input_request: service_pb2.InputRequest) -> service_pb2.EmptyResponse:
        """
        Calls the Input RPC method to send user input to the server.

        Args:
            input_request: An InputRequest protobuf message.

        Returns:
            An EmptyResponse protobuf message.
        """
        try:
            response = self.stub.Input(input_request)
            return response
        except grpc.RpcError as e:
            print(f"Error calling Input: {e}")
            return None  # Or raise the exception

    def decision(self, state: service_pb2.StateResponse) -> service_pb2.InputRequest:
        """
        Determines the input to send based on the current state.

        Args:
            state: The current StateResponse message received from the server.

        Returns:
            An InputRequest protobuf message representing the desired action.

        Note:
            This function currently returns a default input (move forward).
            You should implement your logic here to decide the input based
            on the provided state information (e.g., screen data, velocity).
        """
        # Calculate the total velocity
        total_velocity = self.calculate_velocity(state.linear_velocity)

        # Placeholder logic: Replace this with your actual decision-making process.
        # Example: Always move forward unless velocity exceeds 10.
        if total_velocity > 15:
            forward = False
            reset = True
        else:
            forward = True

        extracted_state = self.process_game_state(state.screen)

        back = False
        left = False
        right = False
        reset = False

        time.sleep(0.2)

        return service_pb2.InputRequest(
            forward=forward,
            back=back,
            left=left,
            right=right,
            reset=reset
        )

    def display_screen(self, screen_bytes):
        """
        Displays the screen image using OpenCV.
        
        Args:
            screen_bytes: The raw bytes of the screen image in RGBA format (1280x720x4).
        """
        try:
            # Convert bytes to numpy array and reshape to (720, 1280, 4)
            nparr = np.frombuffer(screen_bytes, np.uint8)
            img = nparr.reshape((720, 1280, 4))
            
            # Convert RGBA to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Display the image
            cv2.imshow('Marble View', img)
            cv2.waitKey(1)  # Update the window
        except Exception as e:
            print(f"Error displaying screen: {e}")

    def run_interaction_loop(self):
        """
        Runs a loop that repeatedly gets state, determines input, sends input,
        and records the state/input pair.
        """
        while True:
            current_state = self.get_state()
            if current_state is None:
                print("Failed to get state, stopping loop.")
                break

            # Display the current screen
            self.display_screen(current_state.screen)

            # 2. Determine the input based on the state
            input_to_send = self.decision(current_state)

            # Calculate and print total velocity
            total_velocity = self.calculate_velocity(current_state.linear_velocity)
            print(f"Current velocity: {total_velocity:.2f}")

            # 3. Send the input
            response = self.send_input(input_to_send)
            if response is None:
                print("Failed to send input, stopping loop.")
                break

            # 4. Record the state and the input that was sent

            screen_file = os.path.join(self.screen_dir, f"screen_{uuid.uuid4()}")
            recorded_state = {
                'screen': screen_file,
                'linear_velocity': current_state.linear_velocity,
                'total_velocity': total_velocity,  # Added total velocity to recorded state
                'angular_velocity': current_state.angular_velocity,
                'relative_angular_velocity': current_state.relative_angular_velocity,
                'finished': current_state.finished,
                'results': current_state.results,
            }
            with open(screen_file, 'wb') as f:
                f.write(current_state.screen)

            self.records.append((recorded_state, input_to_send))
            if current_state.finished:
                for index, result in enumerate(current_state.results):
                    if result.name == self.name:
                        # Assuming result.name is a string, adjust as necessary
                        print(f"Result {index}: {result.name}, Finish Time: {result.finish_time}, "
                              f"Last Touched Road ID: {result.last_touched_road_id}, "
                              f"Last Touched Road Time: {result.last_touched_road_time}")
                print("Marble finished, stopping loop.")
                break

        print("Interaction loop finished.")

    def calculate_velocity(self, linear_velocity) -> float:
        """
        Calculates the total velocity from linear velocity components.

        Args:
            linear_velocity: A protobuf message containing x, y, z components of linear velocity.

        Returns:
            float: The total velocity magnitude.
        """
        return (linear_velocity.x ** 2 + linear_velocity.y ** 2 + linear_velocity.z ** 2) ** 0.5
    
    def classify_segments(self, segments: list) -> list:
        """
        Classifies the segments of the game state.
        """
        return segments

    def process_game_state(self, screen_bytes) -> dict:
        """
        Processes the game state from the screen image by cropping and segmenting it.

        Args:
            screen_bytes: The raw bytes of the screen image in RGBA format (1280x720x4).

        Returns:
            list: A list of 4 numpy arrays representing the vertical segments of the processed image.
        """
        # Convert bytes to numpy array and reshape to (720, 1280, 4)
        nparr = np.frombuffer(screen_bytes, np.uint8)
        img = nparr.reshape((720, 1280, 4))
        
        # Convert RGBA to BGR (OpenCV format)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Crop the lower part of the image (keep only the top 80% of the image)
        height, width = img.shape[:2]
        cropped_height = int(height * 0.8)
        cropped_img = img[:cropped_height, :]
        
        # Split the image into 4 vertical segments
        segment_width = width // 4
        segments = []
        for i in range(4):
            start_x = i * segment_width
            end_x = (i + 1) * segment_width
            segment = cropped_img[:, start_x:end_x]
            segments.append(segment)
        
        return {"road_segments_ahead":self.classify_segments(segments)}

    def get_records_as_dataframe(self) -> pd.DataFrame:
        """
        Converts the recorded state/input pairs into a pandas DataFrame.

        Returns:
            A pandas DataFrame containing the recorded interaction history.
        """
        data = []
        for state, input_req in self.records:
            # Helper to handle optional fields in ResultEntry
            def get_optional_float(value):
                return value if value is not None else pd.NA

            def get_optional_uint64(value):
                return value if value is not None else pd.NA

            results_list = []
            if state['results']:
                results_list = [
                    {
                        'name': r.name,
                        'finish_time': get_optional_float(r.finish_time),
                        'last_touched_road_id': get_optional_uint64(r.last_touched_road_id),
                        'last_touched_road_time': get_optional_float(r.last_touched_road_time)
                    } for r in state['results']
                ]

            data.append({
                # State fields
                'screen': state['screen'],  # Keep as bytes, or process further if needed
                'linear_velocity_x': state['linear_velocity'].x,
                'linear_velocity_y': state['linear_velocity'].y,
                'linear_velocity_z': state['linear_velocity'].z,
                'angular_velocity_x': state['angular_velocity'].x,
                'angular_velocity_y': state['angular_velocity'].y,
                'angular_velocity_z': state['angular_velocity'].z,
                'relative_angular_velocity_x': state['relative_angular_velocity'].x,
                'relative_angular_velocity_y': state['relative_angular_velocity'].y,
                'relative_angular_velocity_z': state['relative_angular_velocity'].z,
                'finished': state['finished'],
                'results': results_list,  # Store list of result dicts

                # Input fields
                'input_forward': input_req.forward,
                'input_back': input_req.back,
                'input_left': input_req.left,
                'input_right': input_req.right,
                'input_reset': input_req.reset
            })
        df = pd.DataFrame(data)
        return df

    def close(self):
        """Closes the gRPC channel and OpenCV windows."""
        if self.channel:
            self.channel.close()
            print("gRPC channel closed.")
        cv2.destroyAllWindows()  # Close all OpenCV windows
