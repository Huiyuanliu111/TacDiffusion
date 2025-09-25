import socket
import threading
import time
import json
from helper_functions.DataBuffer_Class import DataBuffer
from act.temporal_agg import temporal_aggregation
from act.policy import ACTPolicy
import torch
import struct

def udp_model_receiver(ip_host, port_host, ip_target, port_target, device, model, config, model_train_timeslot=7):
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
    num_queries = config['num_queries']
    all_time_actions = torch.zeros((max_steps, max_steps + num_queries - 1, y_dim), device=device)
    print("Initialized temporal aggregation state.")
    
    # Initialize state sequence buffer for model input
    num_obs = config['num_obs']
    state_dim = 18  # 假设状态维度为18，可以根据实际情况调整
    state_buffer = torch.zeros((num_obs, state_dim), device=device)
    print(f"Initialized state buffer with shape: {state_buffer.shape}")


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
        nonlocal state_buffer

        while True:
            message = data_buffer.get_data()
            if message is not None:
                # Process data using the model
                with torch.no_grad():
                    # 提取latest_data（前18维），忽略previous_data（后18维）
                    full_data = torch.Tensor(message[0]).type(torch.FloatTensor).to(device)  # message[0]是36维数据
                    latest_data = full_data[:18]  # 只要latest_data（前18维）
                    
                    # 更新状态缓冲区（滑动窗口）
                    state_buffer[:-1] = state_buffer[1:].clone()  # 向左移动
                    state_buffer[-1] = latest_data      # 添加最新的18维状态
                    
                    # 准备模型输入：[batch_size, num_obs, state_dim]
                    qpos = state_buffer.unsqueeze(0)  # [1, num_obs, state_dim]
                    
                    # 创建虚拟图像输入
                    dummy_image = torch.zeros((1, 3, 224, 224)).to(device)
                    
                    # 模型推理
                    y_pred_ = model(qpos, dummy_image)  
                    y_pred_tensor = y_pred_[0]

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
                    # 也重置状态缓冲区
                    state_buffer.zero_()

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

# --- 配置文件加载函数 ---
def load_config_from_json(config_path):
    """从JSON文件加载配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"📂 从配置文件加载参数: {config_path}")
    print(f"✅ 配置参数: {config}")
    return config

# --- Model and Network Configuration ---
model_name = 'best_8_group/group1_best_model.pth'
config_file = 'best_8_group/group1_best_config.json'
print(f'model_name: {model_name}')
print(f'config_file: {config_file}')

# 从JSON文件加载配置
config = load_config_from_json(config_file)

# --- Temporal Aggregation Parameters ---
k = 0.01           # Exponential weight decay factor
y_dim = 6          # Action dimension (e.g., Fx, Fy, Fz, Tx, Ty, Tz)
max_steps = 5000   # Reset state after this many steps to prevent memory overflow

print(f"📝 使用配置中的 num_queries: {config['num_queries']}")
print(f"📝 使用配置中的 num_obs: {config['num_obs']}")

ip_host = "0.0.0.0"  # IP address of the model computer
port_host = 1501

ip_target = "10.157.175.246"  # IP address of the robot computer
port_target = 2333
model_train_timeslot = 7 # ms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# Initialize model
print("Loading PyTorch model...")
model = ACTPolicy(config).to(device)
model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()
print("PyTorch model loaded successfully!")

start_time = time.time()
udp_model_receiver(ip_host, port_host, ip_target, port_target, device, model, config, model_train_timeslot=model_train_timeslot)
