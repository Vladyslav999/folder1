import pandas as pd
import numpy as np
import os
import dask.dataframe as dd


work_dir = os.getcwd()
N_COLS = 256


class Feature_generator:
    
    def __init__(self, train_df, test_df, n_cols, normalization_method, feature_type, save_tsv):
        self.train_df = train_df
        self.test_df = test_df
        self.n_cols = n_cols
        self.n_rows_train = len(self.train_df)
        self.n_rows_test = len(self.test_df)
        self.normalization_method = normalization_method
        self.feature_type = feature_type
        self.save_tsv = save_tsv
        
    def preprocess_data(self, df, train=True):
        """ Extracts int features from df feature column"""
        if train:
            n_rows = self.n_rows_train
        else:
            n_rows = self.n_rows_test
            
        result = np.array([int(y) for z in df['features'].apply(lambda x: x.split(',')).compute()
                           for y in z]).reshape(n_rows, self.n_cols+1)
        
        # get only needed feature type
        result = result[result[:, 0] == self.feature_type][:, 1:]
        if result.any():
            return result
        else:
            raise Exception(f"No available data for the feature_type {feature_type}")
            
    def return_statistics(self):
        result = self.preprocess_data(self.train_df)
        mean_train_df_arr = result.mean(axis=0)
        std_train_df_arr = result.std(axis=0)
        self.mean_train_df_arr = mean_train_df_arr
        self.std_train_df_arr = std_train_df_arr
            
    def normalize_df(self):
        """ Normalize df values"""
        result_test = self.preprocess_data(self.test_df, train=False)
        if self.normalization_method=='standardize':
            result_test = (result_test - self.mean_train_df_arr)/self.std_train_df_arr
        else:
            raise Exception(f"{normalization_method} is not implemented")
        return result_test
    
    
    def get_max_feature_index(self, test_array):
        """ Returns index of max element in a row"""
        max_feature_index_array = test_array.argmax(axis=1)
        return max_feature_index_array

    
    def get_max_feature(self, test_array):
        """ Returns max element in a row"""
        max_feature_array = test_array.max(axis=1)
        return max_feature_array


    def get_max_feature_abs_mean_diff(self):
        """ Returns max_feature_abs_mean_diff """
        result_test = self.preprocess_data(self.test_df, train=False)
        max_index = self.get_max_feature_index(result_test)
        max_value = self.get_max_feature(result_test)
        max_feature_2_abs_mean_diff = max_value - self.mean_train_df_arr[max_index]
        
        return max_feature_2_abs_mean_diff
    
    def generate_test_proc(self):
        self.return_statistics()
        result_test_norm  = self.normalize_df()
        
        res = pd.DataFrame(result_test_norm, columns=[f'feature_{self.feature_type}_stand_{i}' for i in range(result_test_norm.shape[1])])
        res['job_id'] = np.array(self.test_df[self.test_df['features'].apply(lambda x: x.startswith(str(self.feature_type)))]['id_job'])
        res[f'max_feature_{self.feature_type}_index'] = self.get_max_feature_index(result_test_norm)
        res[f'max_feature_{self.feature_type}_abs_mean_diff'] = self.get_max_feature_abs_mean_diff()
        
        if self.save_tsv:
            res.to_csv('test_proc.tsv', sep='\t', index=False)
        
        return res
    
    
if __name__ == "__main__":
    ddf_train = dd.read_csv(os.path.join(work_dir, 'train.tsv'), sep='\t')
    ddf_test = dd.read_csv(os.path.join(work_dir, 'test.tsv'), sep='\t')
    solution = Feature_generator(train_df=ddf_train, test_df=ddf_test, n_cols=N_COLS,
                                 normalization_method='standardize', feature_type=2, save_tsv=True)
    print(solution.generate_test_proc().head())