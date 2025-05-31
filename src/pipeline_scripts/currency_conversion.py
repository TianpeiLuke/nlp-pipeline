import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_currency_code(
    marketplace_id: Union[int, float],
    marketplace_info: Dict[str, Dict[str, str]],
    default_currency: str
) -> str:
    """Get currency code for a given marketplace ID."""
    try:
        if pd.isna(marketplace_id) or str(int(marketplace_id)) not in marketplace_info:
            return default_currency
        return marketplace_info[str(int(marketplace_id))]["currency_code"]
    except (ValueError, TypeError):
        return default_currency


def combine_currency_codes(
    df: pd.DataFrame,
    marketplace_id_col: str,
    currency_col: Optional[str],
    marketplace_info: Dict[str, Dict[str, str]],
    default_currency: str,
    skip_invalid_currencies: bool
) -> Tuple[pd.DataFrame, str]:
    """Combine currency codes from marketplace ID and existing currency column."""
    df["currency_code_from_marketplace_id"] = df[marketplace_id_col].apply(
        lambda x: get_currency_code(x, marketplace_info, default_currency)
    )

    if currency_col and currency_col in df.columns:
        df[currency_col] = df[currency_col].combine_first(df["currency_code_from_marketplace_id"])
        final_currency_col = currency_col
    else:
        final_currency_col = "currency_code_from_marketplace_id"

    # Handle invalid currencies
    if not skip_invalid_currencies:
        df = df.dropna(subset=[final_currency_col]).reset_index(drop=True)
    else:
        # Replace invalid currencies with default
        df[final_currency_col] = df[final_currency_col].fillna(default_currency)

    return df, final_currency_col


def currency_conversion_single_variable(
    args: Tuple[pd.DataFrame, str, pd.Series]
) -> pd.Series:
    """Convert single variable's currency values."""
    df, variable, exchange_rate_series = args
    return df[variable] / exchange_rate_series.values


def parallel_currency_conversion(
    df: pd.DataFrame,
    currency_col: str,
    currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, float],
    n_workers: int = 50
) -> pd.DataFrame:
    """Perform parallel currency conversion on multiple variables."""
    exchange_rate_series = df[currency_col].apply(lambda x: currency_conversion_dict.get(x, 1.0))
    processes = min(cpu_count(), len(currency_conversion_vars), n_workers)
    
    with Pool(processes=processes) as pool:
        results = pool.map(
            currency_conversion_single_variable,
            [(df[[var]], var, exchange_rate_series) for var in currency_conversion_vars]
        )
        df[currency_conversion_vars] = pd.concat(results, axis=1)
    
    return df


def process_currency_conversion(
    df: pd.DataFrame,
    marketplace_id_col: str,
    currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, float],
    marketplace_info: Dict[str, Dict[str, str]],
    currency_col: Optional[str] = None,
    default_currency: str = "USD",
    skip_invalid_currencies: bool = False,
    n_workers: int = 50
) -> pd.DataFrame:
    """Process currency conversion."""
    # Drop rows with missing marketplace IDs
    df = df.dropna(subset=[marketplace_id_col]).reset_index(drop=True)

    # Get and combine currency codes
    df, final_currency_col = combine_currency_codes(
        df,
        marketplace_id_col,
        currency_col,
        marketplace_info,
        default_currency,
        skip_invalid_currencies
    )

    # Filter variables that exist in the DataFrame
    currency_conversion_vars = [
        var for var in currency_conversion_vars if var in df.columns
    ]

    if currency_conversion_vars:
        logger.info(f"Converting currencies for variables: {currency_conversion_vars}")
        df = parallel_currency_conversion(
            df,
            final_currency_col,
            currency_conversion_vars,
            currency_conversion_dict,
            n_workers
        )
        logger.info("Currency conversion completed")
    else:
        logger.warning("No variables require currency conversion")

    return df


def main():
    parser = argparse.ArgumentParser()
    # Processing configuration
    parser.add_argument("--n-workers", type=int, default=50)
    parser.add_argument("--marketplace-id-col", required=True)
    parser.add_argument("--currency-col")
    parser.add_argument("--default-currency", default="USD")
    parser.add_argument("--skip-invalid-currencies", action="store_true")
    parser.add_argument("--enable-conversion", type=lambda x: x.lower() == "true", default=True)
    
    args = parser.parse_args()

    # Get environment variables
    currency_conversion_vars = json.loads(os.environ["CURRENCY_CONVERSION_VARS"])
    currency_conversion_dict = json.loads(os.environ["CURRENCY_CONVERSION_DICT"])
    marketplace_info = json.loads(os.environ["MARKETPLACE_INFO"])

    # SageMaker processing paths
    input_path = "/opt/ml/processing/input/data/data.csv"
    output_path = "/opt/ml/processing/output/processed_data.csv"

    logger.info("Starting currency conversion processing")
    
    try:
        # Read input data
        logger.info(f"Reading input data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Input data shape: {df.shape}")

        # Process currency conversion if enabled
        if args.enable_conversion:
            logger.info("Currency conversion enabled")
            df = process_currency_conversion(
                df=df,
                marketplace_id_col=args.marketplace_id_col,
                currency_conversion_vars=currency_conversion_vars,
                currency_conversion_dict=currency_conversion_dict,
                marketplace_info=marketplace_info,
                currency_col=args.currency_col,
                default_currency=args.default_currency,
                skip_invalid_currencies=args.skip_invalid_currencies,
                n_workers=args.n_workers
            )
        else:
            logger.info("Currency conversion disabled")

        # Save processed data
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
