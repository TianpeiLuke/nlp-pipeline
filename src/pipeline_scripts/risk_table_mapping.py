#!/usr/bin/env python
import argparse
import os
import psutil
import subprocess
import pandas as pd
import numpy as np
import json
import pickle as pkl
from sklearn.impute import SimpleImputer


def load_json_config(config_path):
    with open(config_path, "r") as file:
        config_dict = json.load(file)
    return config_dict


def process_downloaded_data(data_dir, data_type):
    # 1) Unzip "part*.gz"
    os.system(f"gzip -d {data_dir}/part*.gz")

    # 2) Combine shards into {data_type}_data.csv + {data_type}_signature.csv
    data_file_name = f"{data_type}_data.csv"
    data_path = os.path.join(data_dir, data_file_name)
    signature_file_name = f"{data_type}_signature.csv"
    signature_path = os.path.join(data_dir, signature_file_name)

    os.system(
        f"cd {data_dir} && "
        "for file in part*.csv; do sed '1d' $file ; done > {data_file_name} && "
        "head -n1 part-00000*.csv > {signature_file_name} && "
        "sed -i 's/__DOT__/./g' {signature_file_name}"
    )
    os.system(f"rm {data_dir}/part*.csv")
    return data_path, signature_path


class OfflineBinning:
    def __init__(self, df, metadata, tag, is_masked=False):
        self.df = df
        self.target = tag
        # Drop any -1 or NaN target rows
        self.df = self.df.loc[
            (self.df[self.target] != -1) & (~self.df[self.target].isnull())
        ]
        if is_masked:
            msk = np.random.rand(len(self.df)) < 0.9
            self.binning_data = self.df[~msk]
            self.df = self.df[msk]
        else:
            self.binning_data = self.df

        self.risk_tables: dict[str, dict] = {}
        self.metadata = metadata
        # Only consider categorical columns (except "marketplaceCountryCode")
        self.variables = [
            v
            for v in self.df.columns
            if v
            in set(
                self.metadata.loc[
                    self.metadata.iscategory & (self.metadata.varname != "marketplaceCountryCode"),
                    "varname",
                ]
            )
        ]
        self.mapper: list[tuple[list[str], list[Any]]] = []
        return

    def add_riskTable(self, smooth_factor: float = 0, count_threshold: int = 0) -> None:
        default_risk = float(self.df.loc[self.df[self.target] != -1, self.target].mean())
        smooth_samples = int(len(self.df) * smooth_factor)

        for var in self.variables:
            self.risk_tables[var] = {
                "varName": var,
                "type": "categorical",
                "mode": self.metadata.loc[self.metadata.varname == var]["datatype"].iat[0],
                "default_bin": default_risk,
            }

            if self.df[var].isnull().sum() == self.df.shape[0]:
                # All values missing → no bins
                self.risk_tables[var]["bins"] = {}
                continue

            risk_table = self.risk_transform(var, default_risk, smooth_samples, count_threshold)
            self.risk_tables[var]["bins"] = risk_table
            self.mapper.append(([var], [risk_table, default_risk]))

    def risk_transform(
        self,
        variable: str,
        default_risk: float,
        samples: int = 0,
        count_threshold: int = 0
    ) -> dict[Any, float]:
        # Build a crosstab: rows = category, columns = target=0 vs. target=1
        cross_tab = (
            pd.crosstab(
                self.binning_data[variable],
                self.binning_data[self.target].astype(object),
                margins=True,
                margins_name="_count_",
                dropna=False,
            )
            .reset_index()
        )

        # point‐estimate of risk = P(Y=1 | category)
        cross_tab["risk"] = cross_tab.apply(lambda x: x.get(1, 0.0) / (x.get(1, 0) + x.get(0, 0)), axis=1)

        # “smoothed” risk with prior = default_risk
        cross_tab["smooth_risk"] = cross_tab.apply(
            lambda x: (
                ((x["_count_"] * x["risk"] + samples * default_risk) / (x["_count_"] + samples))
                if x["_count_"] >= count_threshold
                else default_risk
            ),
            axis=1,
        )

        # Drop the “_count_” row
        cross_tab = cross_tab.loc[cross_tab[variable] != "_count_"]

        values = cross_tab[variable]
        risk_table = dict(zip(values, cross_tab["smooth_risk"]))

        return risk_table


class MissingValueImputation:
    def __init__(self, df, metadata, tag):
        self.df = df
        self.target = tag
        self.df = self.df.loc[
            (self.df[self.target] != -1) & (~self.df[self.target].isnull())
        ]
        self.metadata = metadata
        # All numeric columns (datatype == "numeric")
        self.numeric_variables = [
            v
            for v in self.df.columns
            if v in set(self.metadata.loc[self.metadata["datatype"] == "numeric", "varname"])
        ]
        self.mapper: list[tuple[list[str], list[Any]]] = []
        return

    def add_imputer(self) -> None:
        for var in self.numeric_variables:
            impute_strategy = self.metadata[self.metadata["varname"] == var]["impute_strategy"].iat[0]
            if impute_strategy != "none":
                imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
                imputer.fit(self.df[var].values.reshape(-1, 1))
                self.mapper.append(([var], [imputer]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)
    args = parser.parse_known_args()[0]
    data_type = args.data_type

    ### Input directories inside the container
    input_data_dir = "/opt/ml/processing/input/data"
    config_dir = "/opt/ml/processing/input/config"
    output_dir = "/opt/ml/processing/output"

    # 1) Load the master config JSON (config.json)
    config_path = os.path.join(config_dir, "config.json")
    config = load_json_config(config_path)

    data_config = config["data_config"]
    data_processing_info = data_config["data_processing_info"]
    data_processing_var_list = data_processing_info["data_processing_var_list"]

    tag = config["tag"]

    # 2) Load model_var_list, model_registration_additional_var_list, additional_var_list
    model_var_list = config["model_var_list"]
    model_registration_additional_var_list = config["model_registration_additional_var_list"]
    additional_var_list = config["additional_var_list"]

    # 3) Possibly load currency conversion table if enabled
    currency_conversion_dict = {}
    enable_currency_conversion = config.get("enable_currency_conversion", False)
    if enable_currency_conversion:
        currency_conversion_table_path = os.path.join(
            config_dir, "currency_conversion_table.csv"
        )
        currency_df = pd.read_csv(currency_conversion_table_path)
        currency_conversion_dict = dict(
            zip(currency_df["currency_code"], currency_df["exchange_rate"])
        )
        config["currency_conversion_table"] = currency_df

    # 4) Update config dict with these lists
    config["model_var_list"] = model_var_list
    config["model_registration_additional_var_list"] = model_registration_additional_var_list
    config["data_processing_var_list"] = data_processing_var_list
    config["additional_var_list"] = additional_var_list
    config["currency_conversion_dict"] = currency_conversion_dict

    # 5) Load metadata.csv
    metadata_path = os.path.join(config_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    config["metadata"] = metadata

    # 6) Determine final_model_var_list by dropping any column that is 100% null
    data_dir = f"{input_data_dir}/{data_type}"
    data_path, signature_path = process_downloaded_data(data_dir, data_type)
    signature = pd.read_csv(signature_path).columns
    data = pd.read_csv(data_path, names=signature)

    null_model_var_list = []
    for var in signature:
        if data[var].isnull().all() and (var in model_var_list):
            null_model_var_list.append(var)

    final_model_var_list = [v for v in model_var_list if v not in null_model_var_list]
    config["null_model_var_list"] = null_model_var_list
    config["final_model_var_list"] = final_model_var_list

    # 7) Build risk tables
    offline_bin = OfflineBinning(data, metadata, tag, is_masked=False)
    category_risk_params = config["model_training_config"]["category_risk_params"]
    offline_bin.add_riskTable(**category_risk_params)

    # 8) Build missing‐value imputers
    missing_value_imputer = MissingValueImputation(data, metadata, tag)
    missing_value_imputer.add_imputer()

    bin_mapper = offline_bin.mapper
    missing_value_imputation_mapper = missing_value_imputer.mapper

    # 9) Extract risk tables into a serializable dict
    list_bin: dict[str, dict[str, Any]] = {}
    for item in bin_mapper:
        var_name = item[0][0]
        mapping_dict = item[1][0]   # dict of value→risk
        default_bin = item[1][1]    # default risk
        list_bin[var_name] = {**mapping_dict, "default_bin": default_bin}

    # 10) Save bin_mapping.pkl
    bin_output_path = os.path.join(output_dir, "bin_mapping.pkl")
    with open(bin_output_path, "wb") as f:
        pkl.dump(list_bin, f)

    # 11) Extract imputation dictionary
    missing_value_impute_dict: dict[str, float] = {}
    for item in missing_value_imputation_mapper:
        var_name = item[0][0]
        imputer = item[1][0]
        fillin_value = imputer.statistics_[0]
        missing_value_impute_dict[var_name] = fillin_value

    # 12) Save missing_value_imputation.pkl
    impute_output_path = os.path.join(output_dir, "missing_value_imputation.pkl")
    with open(impute_output_path, "wb") as f:
        pkl.dump(missing_value_impute_dict, f)

    # 13) Save the expanded config dictionary under config.pkl
    config_output_path = os.path.join(output_dir, "config.pkl")
    with open(config_output_path, "wb") as f:
        pkl.dump(config, f)

    print("Risk‐table mapping complete.")
