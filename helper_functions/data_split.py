import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RobotCustomDataset(Dataset):
    def __init__(
        self, DATASET_PATH, transform=None, data_usage="train", train_prop=0.80, 
        state_dataset='robot_state_training.pkl', action_dataset='robot_action_training.pkl',
        num_queries=400, sample_ratio=1.0, maximum_length=12000, minimum_length=100,
        normalize_data=True, num_obs=1000
    ):
        self.DATASET_PATH = DATASET_PATH
        # Optionally apply transformations to the data
        self.transform = transform
        self.num_queries = num_queries  # 保存num_queries
        self.sample_ratio = sample_ratio
        self.maximum_length = maximum_length
        self.minimum_length = minimum_length
        self.normalize_data = normalize_data
        self.num_obs = num_obs
        
        # Construct the file path for the state dataset pickle file
        pkl_file_path_state = os.path.join(DATASET_PATH, state_dataset)

        # Load state data from the pickle file
        try:
            with open(pkl_file_path_state, 'rb') as f:
                self.state_all_list = pickle.load(f)
                self.state_all_list = [state for state in self.state_all_list if len(state) >= self.minimum_length and len(state) <= self.maximum_length]
            # Print the shape of the loaded state data
            print(f"Total epiosodes: {len(self.state_all_list)}") 

        except FileNotFoundError:
            print(f"Error: Pickle file '{pkl_file_path_state}' not found.")
        except Exception as e:
            print(f"Error: {e}")

        pkl_file_path_action = os.path.join(DATASET_PATH, action_dataset)

        # Load action data from the pickle file
        try:
            with open(pkl_file_path_action, 'rb') as f:
                self.action_all_list = pickle.load(f)
                self.action_all_list = [action for action in self.action_all_list if len(action) >= self.minimum_length and len(action) <= self.maximum_length]
        except FileNotFoundError:
            print(f"Error: Pickle file '{pkl_file_path_action}' not found.")
        except Exception as e:
            print(f"Error: {e}")

        # Set a random seed to ensure reproducibility
        random_seed = 4

        # Randomly split the data into train and validation sets
        state_train, state_valid, action_train, action_valid = train_test_split(
            self.state_all_list, self.action_all_list, train_size=train_prop, random_state=random_seed
        )

        if data_usage == "train":
            # Assign the training data
            self.state_all = state_train
            print(f"random split train state data with length: {len(self.state_all)}")
            self.action_all = action_train
        elif data_usage == "valid":
            # Assign the validation data
            self.state_all = state_valid
            print(f"random split valid state data with length: {len(self.state_all)}")
            self.action_all = action_valid
        elif data_usage == "test":
            # Assign the test data
            # Calculate the number of training samples, the rest data is used as testing data
            n_train = int(len(self.state_all_list) * train_prop)
            self.state_all = self.state_all_list[n_train:]
            print(f"test state data with length: {len(self.state_all)}")
            self.action_all = self.action_all_list[n_train:]
        else:
            raise NotImplementedError

        total_states = sum(len(ep) for ep in self.state_all)
        total_actions = sum(len(ep) for ep in self.action_all)
        state_dim = self.state_all[0][0].shape
        action_dim = self.action_all[0][0].shape
        
        self.states_array = np.zeros((total_states,) + state_dim, dtype=np.float32)
        self.actions_array = np.zeros((total_actions,) + action_dim, dtype=np.float32)
        self.episode_indices = []
        
        state_idx = 0
        action_idx = 0
        
        for state_ep, action_ep in zip(self.state_all, self.action_all):
            state_start = state_idx
            action_start = action_idx
            
            state_ep = np.array(state_ep)
            self.states_array[state_idx:state_idx + len(state_ep)] = state_ep
            state_idx += len(state_ep)
            
            action_ep = np.array(action_ep)
            self.actions_array[action_idx:action_idx + len(action_ep)] = action_ep
            action_idx += len(action_ep)
            
            self.episode_indices.append({
                'state_start': state_start,
                'state_end': state_idx,
                'action_start': action_start,
                'action_end': action_idx
            })

       
        if self.normalize_data:
            self._normalize_data(data_usage)

        if self.sample_ratio < 1.0:
            total_samples = len(self.states_array)
            num_samples = int(total_samples * self.sample_ratio)
            self.sample_indices = np.random.choice(total_samples, num_samples, replace=False)
            self.sample_indices.sort()
        else:
            self.sample_indices = np.arange(len(self.states_array))
            
        # Add these properties for compatibility with test code
        self.state = self.states_array
        self.action = self.actions_array

    def _normalize_data(self, data_usage):
        """对数据进行标准化处理"""
        stats_path = os.path.join(self.DATASET_PATH, 'normalization_stats.pkl')
        
        if data_usage == "train":
            # 训练集：计算并保存统计信息
            print("🔄 计算训练数据的标准化统计信息...")
            
            # 计算action数据的统计信息
            action_mean = np.mean(self.actions_array, axis=0)
            action_std = np.std(self.actions_array, axis=0)
            
            # 计算state数据的统计信息
            state_mean = np.mean(self.states_array, axis=0)
            state_std = np.std(self.states_array, axis=0)
            
            # 保存统计信息
            stats = {
                'action_mean': action_mean,
                'action_std': action_std,
                'state_mean': state_mean,
                'state_std': state_std
            }
            
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            
            
            self.action_mean = action_mean
            self.action_std = action_std
            self.state_mean = state_mean
            self.state_std = state_std
            
        else:
            # 验证集/测试集：加载统计信息
            print("📂 加载训练数据的标准化统计信息...")
            try:
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                
                self.action_mean = stats['action_mean']
                self.action_std = stats['action_std']
                self.state_mean = stats['state_mean']
                self.state_std = stats['state_std']
                
                print(f"✅ 成功加载标准化统计信息")
                
            except FileNotFoundError:
                print(f"❌ 错误：找不到标准化统计文件 {stats_path}")
                print("请先运行训练数据集以生成统计信息！")
                raise
        
        # 应用标准化
        print("🔄 应用数据标准化...")
        
        # 标准化action数据
        # 避免除零错误
        action_std_safe = np.where(self.action_std < 1e-8, 1.0, self.action_std)
        self.actions_array = (self.actions_array - self.action_mean) / action_std_safe
        
        # 标准化state数据
        state_std_safe = np.where(self.state_std < 1e-8, 1.0, self.state_std)
        self.states_array = (self.states_array - self.state_mean) / state_std_safe
        
        print("✅ 数据标准化完成！")

    def denormalize_actions(self, normalized_actions):
        """将标准化后的动作数据还原为原始数据"""
        if not self.normalize_data:
            return normalized_actions
        
        return normalized_actions * self.action_std + self.action_mean
    
    def denormalize_states(self, normalized_states):
        """将标准化后的状态数据还原为原始数据"""
        if not self.normalize_data:
            return normalized_states
        
        return normalized_states * self.state_std + self.state_mean

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        actual_index = self.sample_indices[index]
        
        # Find episode info
        episode_info = None
        for ep_info in self.episode_indices:
            if ep_info['state_start'] <= actual_index < ep_info['state_end']:
                episode_info = ep_info
                break
        
        # Calculate relative position in current episode
        relative_pos = actual_index - episode_info['state_start']
        
        # Initialize state sequence and its padding mask
        state_seq = np.zeros((self.num_obs,) + self.states_array[0].shape, dtype=self.states_array.dtype)
        state_pad = np.zeros(self.num_obs, dtype=bool)  # False means not padded
        
        # Fill in the current state
        state_seq[-1] = self.states_array[actual_index]
        
        # Fill in historical states
        for i in range(self.num_obs - 1):
            history_idx = actual_index - (self.num_obs - 1 - i)
            if history_idx >= episode_info['state_start']:
                # If historical state exists in current episode
                state_seq[i] = self.states_array[history_idx]
            else:
                # If historical state is before episode start, use padding
                state_seq[i] = self.states_array[episode_info['state_start']]  # Use episode's first state as padding
                state_pad[i] = True  # Mark as padded
        
        # Process actions (same as before)
        action_start_idx = episode_info['action_start'] + relative_pos
        action_end_in_episode = episode_info['action_end']
        available_actions = action_end_in_episode - action_start_idx
        
        if available_actions >= self.num_queries:
            action = self.actions_array[action_start_idx:action_start_idx + self.num_queries]
            action_pad = np.zeros(self.num_queries, dtype=bool)
        else:
            action = np.zeros((self.num_queries,) + self.actions_array.shape[1:], dtype=self.actions_array.dtype)
            action_pad = np.zeros(self.num_queries, dtype=bool)
            
            if available_actions > 0:
                action[:available_actions] = self.actions_array[action_start_idx:action_end_in_episode]
                last_action = self.actions_array[action_end_in_episode - 1]
                action[available_actions:] = last_action
                action_pad[available_actions:] = True
            else:
                last_action = self.actions_array[action_end_in_episode - 1]
                action[:] = last_action
                action_pad[:] = True

        if self.transform:
            # Apply any transformations to the state sequence
            state_seq = self.transform(state_seq)
            
        return (state_seq, action, state_pad, action_pad)
