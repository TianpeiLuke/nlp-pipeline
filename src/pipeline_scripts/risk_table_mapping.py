import argparse
import os
import typing
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
    # unzip data files
    os.system(f"gzip -d {data_dir}/part*.gz")

    data_file_name = f"{data_type}_data.csv"
    data_path = os.path.join(data_dir, data_file_name)
    signature_file_name = f"{data_type}_signature.csv"
    signature_path = os.path.join(data_dir, signature_file_name)

    os.system(
        f"cd {data_dir} && for file in part*.csv; do sed '1d' $file ; done > {data_path} && head -n1 part-00000*.csv > {signature_path} && sed -i 's/__DOT__/./g' {signature_path} "
    )
    os.system(f"rm {data_dir}/part*.csv")
    return data_path, signature_path


class OfflineBinning:
    def __init__(self, df, metadata, tag, is_masked=False):
        self.df = df
        self.target = tag
        self.df = self.df.loc[(self.df[self.target] != -1) & (~self.df[self.target].isnull())]
        if is_masked:
            msk = np.random.rand(len(self.df)) < 0.9
            self.binning_data = self.df[~msk]
            self.df = self.df[msk]
        else:
            self.binning_data = self.df

        self.risk_tables: typing.Dict = dict()
        self.metadata = metadata
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
        self.mapper: typing.List[typing.Tuple] = list()
        return

    def add_riskTable(self, smooth_factor: float = 0, count_threshold: int = 0) -> None:
        default_risk = float(self.df.loc[self.df[self.target] != -1, self.target].mean())
        smooth_samples = int(len(self.df) * smooth_factor)
        #         ### for debugging purpose
        #         smooth_samples = 0

        for var in self.variables:
            self.risk_tables[var] = dict()
            self.risk_tables[var]["varName"] = var
            self.risk_tables[var]["type"] = "categorical"
            self.risk_tables[var]["mode"] = self.metadata.loc[self.metadata.varname == var][
                "datatype"
            ]
            self.risk_tables[var]["default_bin"] = default_risk
            if self.df[var].isnull().sum() == self.df.shape[0]:
                self.risk_tables[var]["bins"] = dict()
                continue
            risk_table: typing.Dict = self.risk_transform(
                var, default_risk, smooth_samples, count_threshold
            )
            self.risk_tables[var]["bins"] = risk_table

            self.mapper.append(([var], [risk_table, default_risk]))

    def risk_transform(
        self, variable: str, default_risk: float, samples: int = 0, count_threshold: int = 0
    ) -> typing.Dict:
        cross_tab = pd.crosstab(
            self.binning_data[variable],
            self.binning_data[self.target].astype(object),
            margins=True,
            margins_name="_count_",
            dropna=False,
        ).reset_index()
        cross_tab["risk"] = cross_tab.apply(lambda x: x[1] / (x[1] + x[0]), axis=1)
        cross_tab["smooth_risk"] = cross_tab.apply(
            lambda x: (
                (x["_count_"] * x["risk"] + samples * default_risk) / (x["_count_"] + samples)
                if x["_count_"] >= count_threshold
                else default_risk
            ),
            axis=1,
        )

        cross_tab = cross_tab.loc[cross_tab[variable] != "_count_"]

        values = cross_tab[variable]
        risk_table = dict(zip(values, cross_tab["smooth_risk"]))

        return risk_table


class MissingValueImputation:
    def __init__(self, df, metadata, tag):
        self.df = df
        self.target = tag
        self.df = self.df.loc[(self.df[self.target] != -1) & (~self.df[self.target].isnull())]

        self.metadata = metadata
        self.numeric_variables = [
            v
            for v in self.df.columns
            if v
            in set(
                self.metadata.loc[
                    self.metadata["datatype"] == "numeric",
                    "varname",
                ]
            )
        ]
        self.mapper: typing.List[typing.Tuple] = list()
        return

    def add_imputer(self) -> None:
        for var in self.numeric_variables:
            impute_strategy = self.metadata[self.metadata["varname"] == var][
                "impute_strategy"
            ].values[0]
            if impute_strategy != "none":
                imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
                imputer.fit(self.df[var].values.reshape(-1, 1))
                self.mapper.append(([var], [imputer]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str)
    args, _ = parser.parse_known_args()
    data_type = args.data_type

    ### Input ########################################
    input_data_dir = "/opt/ml/processing/input"
    config_dir = f"{input_data_dir}/config"
    output_dir = "/opt/ml/processing/output"

    print("Reading config. ")
    config_path = f"{config_dir}/config.json"
    config = load_json_config(config_path)

    print("Reading data_config. ")
    data_config = config["data_config"]

    data_processing_info = data_config["data_processing_info"]
    data_processing_var_list = data_processing_info["data_processing_var_list"]

    tag = config["tag"]

    print("Reading other configs. ")
    model_var_list = config["model_var_list"]

    model_registration_additional_var_list = config["model_registration_additional_var_list"]

    additional_var_list = config["additional_var_list"]

    currency_conversion_dict = dict()
    enable_currency_conversion = config["enable_currency_conversion"]
    if enable_currency_conversion:
        currency_conversion_table_path = os.path.join(config_dir, "currency_conversion_table.csv")
        currency_conversion_table = pd.read_csv(currency_conversion_table_path)
        currency_conversion_dict = dict(
            zip(
                currency_conversion_table["currency_code"],
                currency_conversion_table["exchange_rate"],
            )
        )
        config["currency_conversion_table"] = currency_conversion_table

    config["model_var_list"] = model_var_list
    config["model_registration_additional_var_list"] = model_registration_additional_var_list
    config["data_processing_var_list"] = data_processing_var_list

    config["currency_conversion_dict"] = currency_conversion_dict

    print("Reading model_training_config data. ")
    model_training_config = config["model_training_config"]
    category_risk_params = model_training_config["category_risk_params"]

    print("Reading metadata. ")
    metadata_path = os.path.join(config_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    config["metadata"] = metadata
    print("metadata shape: ", metadata.shape)

    catvars = []
    for var in data_processing_var_list:
        if var in metadata and var in set(
            metadata.loc[(metadata["iscategory"].astype(bool)), "varname"]
        ):

            catvars.append(var)

    data_dir = f"{input_data_dir}/{data_type}"
    print(f"Reading {data_type} data from {data_dir}")
    print(f"Memory usage: {dict(psutil.virtual_memory()._asdict())}")
    print(f"data directory size: {subprocess.run(['du', '-sh', data_dir])}")

    data_path, signature_path = process_downloaded_data(data_dir, data_type)
    print(f"data_path: {data_path}")
    print(f"signature_path: {signature_path}")
    signature = pd.read_csv(signature_path).columns
    print(f"{data_type} signature: {signature}")
    data = pd.read_csv(data_path, names=signature)
    print(f"{data_type} data shape: {data.shape}")

    null_model_var_list = []
    for var in signature:
        null_pct = data[var].isnull().sum() / data.shape[0]
        print(f"{var} null: {null_pct}")
        if null_pct == 1 and var in model_var_list:
            null_model_var_list.append(var)

    final_model_var_list = [var for var in model_var_list if var not in null_model_var_list]
    config["null_model_var_list"] = null_model_var_list
    config["final_model_var_list"] = final_model_var_list

    offline_bin = OfflineBinning(data, metadata, tag, is_masked=False)
    offline_bin.add_riskTable(**category_risk_params)

    missing_value_imputer = MissingValueImputation(data, metadata, tag)
    missing_value_imputer.add_imputer()

    bin_mapper = offline_bin.mapper
    missing_value_imputation_mapper = missing_value_imputer.mapper

    # get list_bin to save
    list_bin = dict()
    for item in bin_mapper:
        var_name = item[0][0]
        list_bin[var_name] = {key: value for key, value in item[1][0].items()}
        list_bin[var_name]["default_bin"] = item[1][1]

    print("saved bin_mapping!")
    output_location = f"{output_dir}/bin_mapping.pkl"
    pkl.dump(list_bin, open(output_location, "wb"))

    # get missing_value_impute_dict to save
    missing_value_impute_dict = dict()
    for item in missing_value_imputation_mapper:
        var_name = item[0][0]
        imputer = item[1][0]

        fillin_value = imputer.statistics_[0]
        missing_value_impute_dict[var_name] = fillin_value

    output_location = f"{output_dir}/missing_value_imputation.pkl"
    # save missing value imputation mapper
    pkl.dump(
        missing_value_impute_dict,
        open(output_location, "wb"),
    )

    pkl.dump(config, open(f"{output_dir}/config.pkl", "wb"))
