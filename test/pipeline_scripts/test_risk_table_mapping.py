import unittest
from unittest.mock import patch
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import pickle as pkl
from pathlib import Path


# Import the components to be tested
from src.pipeline_scripts.risk_table_mapping import (
    OfflineBinning,
    MissingValueImputation,
    main as risk_mapping_main
)

class TestOfflineBinning(unittest.TestCase):
    """Tests the OfflineBinning class."""
    def setUp(self):
        self.metadata = pd.DataFrame({
            'varname': ['cat_var1', 'cat_var2', 'num_var', 'target'],
            'iscategory': [True, True, False, False],
            'datatype': ['categorical', 'categorical', 'numeric', 'numeric']
        })
        self.df = pd.DataFrame({
            'cat_var1': ['A', 'B', 'A', 'C', 'B'],
            'cat_var2': ['X', 'X', 'Y', 'Y', 'Z'],
            'num_var': [10, 20, 30, 40, 50],
            'target': [1, 0, 1, 0, 1]
        })
        self.tag = 'target'
        self.binner = OfflineBinning(self.metadata, self.tag)

    def test_fit_creates_risk_tables(self):
        """Test that fitting creates the expected risk table structure."""
        self.binner.fit(self.df)
        self.assertIn('cat_var1', self.binner.risk_tables)
        self.assertIn('cat_var2', self.binner.risk_tables)
        self.assertNotIn('num_var', self.binner.risk_tables) # Should ignore numeric vars
        
        # Check content of a risk table
        cat1_bins = self.binner.risk_tables['cat_var1']['bins']
        self.assertIn('A', cat1_bins)
        self.assertAlmostEqual(cat1_bins['A'], 1.0) # 2 events / 2 total
        self.assertAlmostEqual(cat1_bins['B'], 0.5) # 1 event / 2 total

    def test_transform_maps_values(self):
        """Test that transform correctly maps categorical values to risk scores."""
        self.binner.fit(self.df)
        transformed_df = self.binner.transform(self.df)
        
        # Check if values are replaced by their risk scores
        self.assertNotEqual(transformed_df['cat_var1'].iloc[0], 'A')
        self.assertAlmostEqual(transformed_df['cat_var1'].iloc[0], 1.0) # Risk of 'A'
        self.assertAlmostEqual(transformed_df['cat_var1'].iloc[1], 0.5) # Risk of 'B'
        
        # Check that unseen values are mapped to the default risk
        test_df_unseen = pd.DataFrame({'cat_var1': ['D']})
        transformed_unseen = self.binner.transform(test_df_unseen)
        default_risk = self.binner.risk_tables['cat_var1']['default_bin']
        self.assertAlmostEqual(transformed_unseen['cat_var1'].iloc[0], default_risk)

class TestMissingValueImputation(unittest.TestCase):
    """Tests the MissingValueImputation class."""
    def setUp(self):
        self.metadata = pd.DataFrame({
            'varname': ['num_var1', 'num_var2', 'cat_var'],
            'datatype': ['numeric', 'numeric', 'categorical'],
            'impute_strategy': ['mean', 'median', 'none']
        })
        self.df = pd.DataFrame({
            'num_var1': [10, 20, np.nan, 40],
            'num_var2': [100, np.nan, 300, 400],
            'cat_var': ['A', 'B', 'A', 'C']
        })
        self.imputer = MissingValueImputation(self.metadata)

    def test_fit_and_transform(self):
        """Test that fitting creates imputers and transform fills NaNs."""
        self.imputer.fit(self.df)
        self.assertIn('num_var1', self.imputer.imputers)
        self.assertIn('num_var2', self.imputer.imputers)
        
        transformed_df = self.imputer.transform(self.df)
        
        # Check if NaNs are filled
        self.assertFalse(transformed_df['num_var1'].hasnans)
        self.assertFalse(transformed_df['num_var2'].hasnans)
        
        # Check filled values
        self.assertAlmostEqual(transformed_df['num_var1'].iloc[2], (10 + 20 + 40) / 3) # mean
        self.assertAlmostEqual(transformed_df['num_var2'].iloc[1], np.median([100, 300, 400])) # median

class TestMainRiskTableFlow(unittest.TestCase):
    """Tests the main execution flow of the risk table mapping script."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create dummy config and metadata
        self.config = {
            "tag": "target",
            "model_training_config": {"category_risk_params": {}}
        }
        self.metadata = pd.DataFrame({
            'varname': ['cat_var', 'num_var', 'target'],
            'iscategory': [True, False, False],
            'datatype': ['categorical', 'numeric', 'numeric'],
            'impute_strategy': ['none', 'mean', 'none']
        })
        self.config['metadata'] = self.metadata.to_dict()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_main_training_mode(self):
        """Test the main logic in 'training' mode."""
        # Create a single unsplit data file
        df = pd.DataFrame({
            'cat_var': ['A', 'B'] * 50,
            'num_var': list(range(100)),
            'target': [0, 1] * 50
        })
        df.to_csv(os.path.join(self.input_dir, 'data.csv'), index=False)

        # Run main function
        risk_mapping_main(
            job_type='training',
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=self.config,
            train_ratio=0.7,
            test_val_ratio=0.5
        )

        # Assertions
        train_path = os.path.join(self.output_dir, 'train', 'train_processed_data.csv')
        test_path = os.path.join(self.output_dir, 'test', 'test_processed_data.csv')
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))

        # Check that artifacts were saved
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'bin_mapping.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'missing_value_imputation.pkl')))
        
        # Check content of transformed data
        train_df = pd.read_csv(train_path)
        self.assertEqual(len(train_df), 70) # 70% of 100
        self.assertIn('cat_var', train_df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(train_df['cat_var'])) # Should be numeric after risk mapping

    def test_main_inference_mode(self):
        """Test the main logic in a non-training ('validation') mode."""
        # Create pre-split data
        for split in ['train', 'validation', 'test']:
            split_dir = os.path.join(self.input_dir, split)
            os.makedirs(split_dir)
            df = pd.DataFrame({
                'cat_var': ['A', 'B', 'C'],
                'target': [1, 0, 1]
            })
            df.to_csv(os.path.join(split_dir, f'{split}_processed_data.csv'), index=False)
            
        # Run main function
        risk_mapping_main(
            job_type='validation',
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=self.config,
            train_ratio=0.7,
            test_val_ratio=0.5
        )
        
        # Assertions
        val_output_path = os.path.join(self.output_dir, 'validation', 'validation_processed_data.csv')
        self.assertTrue(os.path.exists(val_output_path))
        
        # Check that the validation data was transformed based on the train data
        with open(os.path.join(self.output_dir, 'bin_mapping.pkl'), 'rb') as f:
            bins = pkl.load(f)
        
        # Risk for 'A' in train is 1.0. Check if this is applied to validation.
        val_df = pd.read_csv(val_output_path)
        expected_risk_A = bins['cat_var']['bins']['A']
        self.assertAlmostEqual(val_df.loc[0, 'cat_var'], expected_risk_A)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
