import socket
import threading
import time
from helper_functions.DataBuffer_Class import DataBuffer
from act.temporal_agg import temporal_aggregation
import onnxruntime as ort
import torch
import struct

def udp_model_receiver(ip_host, port_host, ip_target, port_target, device, ort_session, model_train_timeslot=7):
    """
    Function to receive data via UDP, process it with a model, and send the results back via UDP.
    
    Args:
        ip_host (str): IP address for receiving data.
        port_host (int): Port for receiving data.
        ip_target (str): IP address for sending processed data.
        port_target (int): Port for sending processed data.
        device (str): Device to use for PyTorch (e.g., 'cuda' or 'cpu').
        ort_session (onnxruntime.InferenceSession): ONNX Runtime inference session.
        model_train_timeslot (int): Time slot for the model training, in milliseconds.
    """
    
    # Define UDP sockets for sending and receiving
    udp_socket_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket_receive.bind((ip_host, port_host))

    print(f"Listening on {ip_host}:{port_host}...")
    print(f"Sending to {ip_target}:{port_target}...")

    data_buffer = DataBuffer()

    data_buffer.horizon_prev = model_train_timeslot
    print(f'**** data_buffer.horizon_prev: {data_buffer.horizon_prev}ms ****')

    send_count = 0  # Count the number of packets sent

    # Initialize state for temporal aggregation
    loop_counter = 0
    # Pre-allocate a large tensor to store action history for aggregation
    # The size is determined by max_steps, which is the number of steps before resetting the state
    all_time_actions = torch.zeros((max_steps, max_steps + num_queries - 1, y_dim), device=device)
    print("Initialized temporal aggregation state.")


    def receive_data():
        """
        Continuously receive data from the UDP socket, process it, and store it in the buffer.
        """
        while True:
            try:
                data, addr = udp_socket_receive.recvfrom(1024)
                # print(f'Received message in original: {data}')

                if data == b'end':
                    print('Finished receiving data!')
                    data_buffer.empty_buffer()
                else:
                    # Decode bytes to string and parse the string into a list of floats
                    data_str = data.decode('utf-8')
                    data_str = data_str.strip('[]')  # Remove leading and trailing brackets
                    message = [float(x) for x in data_str.split(',')]
                    data_buffer.add_data(message)

            except Exception as e:
                print(f"Error receiving data: {e}")

    def send_data():
        """
        Continuously process data from the buffer using the model and send it via UDP.
        """
        nonlocal send_count
        nonlocal loop_counter
        nonlocal all_time_actions

        while True:
            message = data_buffer.get_data()
            if message is not None:
                # Process data using the model
                with torch.no_grad():
                    x_eval = torch.Tensor(message).type(torch.FloatTensor).to(device)
                    x_eval_ = x_eval.repeat(1, 1).cpu().numpy()
                    y_pred_ = ort_session.run(['output'], {'qpos': x_eval_})[0]
                    y_pred_tensor = torch.from_numpy(y_pred_[0]).to(device)

                    # Perform temporal aggregation
                    raw_action = temporal_aggregation(
                        loop_counter, y_pred_tensor, all_time_actions, num_queries, k
                    )
                
                # Increment and reset counter to prevent overflow and memory leak
                loop_counter += 1
                if loop_counter >= max_steps:
                    print("Max steps reached, resetting temporal aggregation state.")
                    loop_counter = 0
                    all_time_actions.zero_()

                # Prepare data for sending
                payload = raw_action.cpu().flatten().tolist()
                counter = 0
                format_str = "<6b" + str(len(payload)) + "f4b"  # Format string for struct packing
                data_to_send = struct.pack(format_str, 127, 127, 127, 127, counter, len(payload) * 4, *payload, 126, 126, 126, 126)
                udp_socket_send.sendto(data_to_send, (ip_target, port_target))
                send_count += 1  # Increment the count of sent packets

    def print_send_rate():
        """
        Print the rate of data packets sent per second.
        """
        nonlocal send_count
        while True:
            time.sleep(1)
            print(f"Data sent in the last second: {send_count} packets")
            send_count = 0  # Reset the counter

    # Start the data receiving thread
    receive_thread = threading.Thread(target=receive_data)
    receive_thread.daemon = True  # Ensure the thread exits when the main thread exits
    receive_thread.start()

    # Start the send rate printing thread
    rate_thread = threading.Thread(target=print_send_rate)
    rate_thread.daemon = True  # Ensure the thread exits when the main thread exits
    rate_thread.start()

    # Keep the main thread running to allow receiving and sending threads to continue working
    try:
        send_data()
    except KeyboardInterrupt:
        print("Receiver stopped.")
    finally:
        udp_socket_send.close()
        udp_socket_receive.close()

# --- Temporal Aggregation Parameters ---
num_queries = 200  # Action sequence length from the model, MUST match model output
k = 0.01           # Exponential weight decay factor
y_dim = 6          # Action dimension (e.g., Fx, Fy, Fz, Tx, Ty, Tz)
max_steps = 5000   # Reset state after this many steps to prevent memory overflow

# --- Model and Network Configuration ---
model_name = 'output/ACT.onnx'

ip_host = "0.0.0.0"  # IP address of the model computer
port_host = 1501

ip_target = "10.157.175.246"  # IP address of the robot computer
port_target = 2333
model_train_timeslot = 7 # ms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession(model_name, sess_options)
start_time = time.time()
udp_model_receiver(ip_host, port_host, ip_target, port_target, device, ort_session, model_train_timeslot=model_train_timeslot)
