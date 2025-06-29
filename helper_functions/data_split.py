import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RobotCustomDataset(Dataset):
    def __init__(
        self, DATASET_PATH, transform=None, data_usage="train", train_prop=0.80, 
        state_dataset='robot_state_training.pkl', action_dataset='robot_action_training.pkl',
        num_queries=400  # 添加num_queries参数
    ):
        self.DATASET_PATH = DATASET_PATH
        # Optionally apply transformations to the data
        self.transform = transform
        self.num_queries = num_queries  # 保存num_queries
        # Construct the file path for the state dataset pickle file
        pkl_file_path_state = os.path.join(DATASET_PATH, state_dataset)

        # Load state data from the pickle file
        try:
            with open(pkl_file_path_state, 'rb') as f:
                self.state_all = pickle.load(f)
            # Print the shape of the loaded state data
            print(f"Total epiosodes: {len(self.state_all)}") 
            for i in range(3):
                print(f"State Episode {i} has {self.state_all[i].shape} ")
        except FileNotFoundError:
            print(f"Error: Pickle file '{pkl_file_path_state}' not found.")
        except Exception as e:
            print(f"Error: {e}")

        # Construct the file path for the action dataset pickle file
        pkl_file_path_action = os.path.join(DATASET_PATH, action_dataset)

        # Load action data from the pickle file
        try:
            with open(pkl_file_path_action, 'rb') as f:
                self.action_all = pickle.load(f)
            # Print the shape of the loaded action data 
            for i in range(3):
                print(f"Action Episode {i} has {self.action_all[i].shape} ")   
        except FileNotFoundError:
            print(f"Error: Pickle file '{pkl_file_path_action}' not found.")
        except Exception as e:
            print(f"Error: {e}")

        # Set a random seed to ensure reproducibility
        random_seed = 42

        # Randomly split the data into train and validation sets
        state_train, state_valid, action_train, action_valid = train_test_split(
            self.state_all, self.action_all, train_size=train_prop, random_state=random_seed
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

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.state_all)

    def __getitem__(self, index):
        # Retrieve a sample from the dataset at the specified index
        state_episode = self.state_all[index]
        action_episode = self.action_all[index]

        time_step = np.random.randint(0, len(state_episode))
        state = state_episode[time_step]
        action = action_episode[time_step:time_step+self.num_queries]
        
        is_pad = np.zeros(self.num_queries, dtype=bool)
        
        if len(action) < self.num_queries:
            # 如果剩余动作不足，用当前episode的最后一个动作填充
            original_length = len(action)
            last_action = action_episode[-1]
            num_padding = self.num_queries - len(action)
            padding = np.tile(last_action, (num_padding, 1))
            action = np.vstack([action, padding])
            
            # 标记填充的位置（从原始长度开始到结尾都是填充）
            is_pad[original_length:] = True

        if self.transform:
            # Apply any transformations to the state data
            state = self.transform(state)
        return (state, action, is_pad)
