import unittest
import pandas as pd
import numpy as np

from src.processing.numerical_imputation_processor import NumericalVariableImputationProcessor

class TestNumericalVariableImputationProcessor(unittest.TestCase):
    
    def setUp(self):
        # Create test data with explicit index for better debugging
        self.data = pd.DataFrame({
            'age': [25.0, np.nan, 35.0, 40.0, np.nan],
            'income': [60000.0, 70000.0, np.nan, 80000.0, 90000.0],
            'text_col': ['a', 'b', 'c', 'd', 'e']
        }, index=[0, 1, 2, 3, 4])
        
        self.single_value_data = pd.DataFrame({
            'age': [30.0] * 5,
            'income': [50000.0] * 5
        })
        
        self.all_nan_data = pd.DataFrame({
            'age': [np.nan] * 3,
            'income': [np.nan] * 3
        })

    def test_initialization_defaults(self):
        processor = NumericalVariableImputationProcessor(variables=['age', 'income'])
        self.assertEqual(processor.variables, ['age', 'income'])
        self.assertIsNone(processor.imputation_dict)
        self.assertEqual(processor.strategy, 'mean')
        self.assertFalse(processor.is_fitted)

    def test_initialization_custom(self):
        imputation_dict = {'age': 30, 'income': 50000}
        processor = NumericalVariableImputationProcessor(
            variables=['age', 'income'],
            imputation_dict=imputation_dict,
            strategy='median'
        )
        self.assertEqual(processor.variables, ['age', 'income'])
        self.assertEqual(processor.imputation_dict, imputation_dict)
        self.assertEqual(processor.strategy, 'median')
        self.assertTrue(processor.is_fitted)

    def test_fit_mean_strategy(self):
        processor = NumericalVariableImputationProcessor(
            variables=['age', 'income'],
            strategy='mean'
        )
        processor.fit(self.data)
        
        self.assertTrue(processor.is_fitted)
        self.assertAlmostEqual(processor.imputation_dict['age'], self.data['age'].mean())
        self.assertAlmostEqual(processor.imputation_dict['income'], self.data['income'].mean())

    def test_fit_median_strategy(self):
        processor = NumericalVariableImputationProcessor(
            variables=['age', 'income'],
            strategy='median'
        )
        processor.fit(self.data)
        
        self.assertTrue(processor.is_fitted)
        self.assertAlmostEqual(processor.imputation_dict['age'], self.data['age'].median())
        self.assertAlmostEqual(processor.imputation_dict['income'], self.data['income'].median())

    def test_process_single_record(self):
        processor = NumericalVariableImputationProcessor(
            imputation_dict={'age': 30, 'income': 50000}
        )
        self.assertTrue(processor.is_fitted)
        
        input_record = {'age': np.nan, 'income': 75000, 'text_col': 'test'}
        result = processor.process(input_record)
        
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['income'], 75000)
        self.assertEqual(result['text_col'], 'test')

    def test_transform_dataframe(self):
        processor = NumericalVariableImputationProcessor(
            imputation_dict={'age': 30.0, 'income': 50000.0}
        )
        self.assertTrue(processor.is_fitted)
        
        result_df = processor.transform(self.data)
        
        # Check imputation counts
        age_imputed_count = (result_df['age'] == 30.0).sum()
        income_imputed_count = (result_df['income'] == 50000.0).sum()
        
        # Basic checks
        self.assertFalse(result_df['age'].isna().any(), "There should be no NaN values in age column")
        self.assertFalse(result_df['income'].isna().any(), "There should be no NaN values in income column")
        
        # Check imputation counts
        self.assertEqual(age_imputed_count, 2, "Two age values should be imputed with 30.0")
        self.assertEqual(income_imputed_count, 1, "One income value should be imputed with 50000.0")
        
        # Check that original values were preserved
        original_ages = [25.0, 35.0, 40.0]
        original_incomes = [60000.0, 70000.0, 80000.0, 90000.0]
        
        self.assertTrue(all(val in result_df['age'].values for val in original_ages),
                       "Original age values should be preserved")
        self.assertTrue(all(val in result_df['income'].values for val in original_incomes),
                       "Original income values should be preserved")

    def test_transform_series(self):
        processor = NumericalVariableImputationProcessor(
            imputation_dict={'age': 30}
        )
        
        series = pd.Series([25, np.nan, 35], name='age')
        result_series = processor.transform(series)
        
        self.assertFalse(result_series.isna().any())
        self.assertEqual(result_series[1], 30)

    def test_pipeline_compatibility(self):
        processor1 = NumericalVariableImputationProcessor(
            imputation_dict={'age': 30}
        )
        processor2 = NumericalVariableImputationProcessor(
            imputation_dict={'income': 50000}
        )
        
        self.assertTrue(processor1.is_fitted)
        self.assertTrue(processor2.is_fitted)
        
        pipeline = processor1 >> processor2
        
        input_record = {'age': np.nan, 'income': np.nan}
        result = pipeline.process(input_record)
        
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['income'], 50000)

    def test_auto_detect_numeric_columns(self):
        processor = NumericalVariableImputationProcessor(strategy='mean')
        processor.fit(self.data)
        
        self.assertTrue('age' in processor.variables)
        self.assertTrue('income' in processor.variables)
        self.assertFalse('text_col' in processor.variables)

    def test_transform_maintains_non_imputed_columns(self):
        processor = NumericalVariableImputationProcessor(
            imputation_dict={'age': 30}
        )
        
        df = pd.DataFrame({
            'age': [25, np.nan],
            'other': ['a', 'b']
        })
        
        result = processor.transform(df)
        
        self.assertTrue('other' in result.columns)
        pd.testing.assert_series_equal(df['other'], result['other'])

if __name__ == '__main__':
    unittest.main()
