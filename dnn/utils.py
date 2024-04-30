import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None):
        np.random.seed(seed)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        
        # Calculate number of batches
        self.num_batches = self.num_samples // batch_size
        if self.num_samples % batch_size != 0:
            self.num_batches += 1
        
        # For shuffle
        self.index_array = np.arange(self.num_samples)
        self.current_batch = 0
        
        if self.shuffle:
            np.random.shuffle(self.index_array)
    
    def __iter__(self):
        self.current_batch = 0  # Reset current batch
        if self.shuffle:
            np.random.shuffle(self.index_array)
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
        batch_indices = self.index_array[start_idx:end_idx]
        
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        self.current_batch += 1
        
        return batch_X, batch_y
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, item):
        if item >= self.num_batches:
            raise IndexError
        start_idx = item * self.batch_size
        end_idx = min((item + 1) * self.batch_size, self.num_samples)
        batch_indices = self.index_array[start_idx:end_idx]
        
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        return batch_X, batch_y
    