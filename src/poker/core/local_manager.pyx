import os
import pickle


cdef class LocalManager:
    def __init__(self, base_path = 'dat/_tmp'):
        self.base_path = base_path
        self.load()
    
    def get_regret_sum(self):
        return self.regret_sum
    
    def get_strategy_sum(self):
        return self.strategy_sum

    def merge_dicts(self, hash_table, shared_dict):
        for key, value in shared_dict.items():
            hash_table.set(key, value)

    def save(self):
        # print(f"Saving to {self.base_path}")
        os.makedirs(self.base_path, exist_ok=True)
        
        self.regret_sum.dump(self.base_path, 'regret_sum')
        self.strategy_sum.dump(self.base_path, 'strategy_sum')
    
    def load(self):
        print(f"Loading from {self.base_path}.")
        try:
            with open(self.base_path + '/regret_sum.pkl', 'rb') as f:
                self.regret_sum = HashTable(pickle.load(f))
        except Exception as e:
            print(f"Error loading regret sum: {e}")
            self.regret_sum = HashTable({})

        try:
            with open(self.base_path + '/regret_sum.pkl', 'rb') as f:
                self.strategy_sum = HashTable(pickle.load(f))
        except Exception as e:
            print(f"Error loading regret sum: {e}")
            self.strategy_sum = HashTable({})
