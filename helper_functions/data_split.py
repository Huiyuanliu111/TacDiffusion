import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RobotCustomDataset(Dataset):
    def __init__(
        self, DATASET_PATH, transform=None, data_usage="train", train_prop=0.80, 
        state_dataset='robot_state_training.pkl', action_dataset='robot_action_training.pkl',
        num_queries=400, sample_ratio=1.0
    ):
        self.DATASET_PATH = DATASET_PATH
        # Optionally apply transformations to the data
        self.transform = transform
        self.num_queries = num_queries  # 保存num_queries
        self.sample_ratio = sample_ratio
        # Construct the file path for the state dataset pickle file
        pkl_file_path_state = os.path.join(DATASET_PATH, state_dataset)

        # Load state data from the pickle file
        try:
            with open(pkl_file_path_state, 'rb') as f:
                self.state_all_list = pickle.load(f)
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
        except FileNotFoundError:
            print(f"Error: Pickle file '{pkl_file_path_action}' not found.")
        except Exception as e:
            print(f"Error: {e}")

        # Set a random seed to ensure reproducibility
        random_seed = 42

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
            n_train = int(len(self.state_all) * train_prop)
            self.state_all = self.state_all[n_train:]
            print(f"test state data with length: {len(self.state_all)}")
            self.action_all = self.action_all[n_train:]
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

        if self.sample_ratio < 1.0:
            total_samples = len(self.states_array)
            num_samples = int(total_samples * self.sample_ratio)
            self.sample_indices = np.random.choice(total_samples, num_samples, replace=False)
            self.sample_indices.sort()
        else:
            self.sample_indices = np.arange(len(self.states_array))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        actual_index = self.sample_indices[index]
        state = self.states_array[actual_index]
        
        episode_info = None
        for ep_info in self.episode_indices:
            if ep_info['state_start'] <= actual_index < ep_info['state_end']:
                episode_info = ep_info
                break
        
        relative_pos = actual_index - episode_info['state_start']
        action_start_idx = episode_info['action_start'] + relative_pos
        action_end_in_episode = episode_info['action_end']
        available_actions = action_end_in_episode - action_start_idx
        
        if available_actions >= self.num_queries:
            action = self.actions_array[action_start_idx:action_start_idx + self.num_queries]
            is_pad = np.zeros(self.num_queries, dtype=bool)
        else:
            action = np.zeros((self.num_queries,) + self.actions_array.shape[1:], dtype=self.actions_array.dtype)
            is_pad = np.zeros(self.num_queries, dtype=bool)
            
            if available_actions > 0:
                action[:available_actions] = self.actions_array[action_start_idx:action_end_in_episode]
                last_action = self.actions_array[action_end_in_episode - 1]
                action[available_actions:] = last_action
                is_pad[available_actions:] = True
            else:
                last_action = self.actions_array[action_end_in_episode - 1]
                action[:] = last_action
                is_pad[:] = True

        if self.transform:
            # Apply any transformations to the state data
            state = self.transform(state)
        return (state, action, is_pad)
