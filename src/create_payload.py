import os
import tarfile
import pickle as pkl


if __name__ == "__main__":
    artifacts_dir = "/opt/ml/processing/input/artifacts"
    payload_dir = "/opt/ml/processing/mims_payload"

    config_path = f"{artifacts_dir}/config.pkl"
    with open(config_path, "rb") as file:
        config = pkl.load(file)

    data_config = config["data_config"]
    marketplace_id_col = config["marketplace_id_col"]

    model_registration_info = data_config["model_registration_info"]
    registration_step_key_list = list(set(model_registration_info["step_name"].keys()))
    # use the first step_name key
    step_name_key = registration_step_key_list[0]
    model_registration_config = model_registration_info["model_registration_config"][step_name_key]

    var_type = model_registration_config["source_model_inference_input_variable_list"]

    print("create payload file")

    default_numeric_value = "0"
    default_text_value = "MY_TEXT"
    payload_value_dict = {
        marketplace_id_col: "3",
    }

    payload_value_list = []
    for var in var_type:
        if var in payload_value_dict:
            payload_value_list.append(payload_value_dict[var])
        else:
            if var_type[var] == "NUMERIC":
                payload_value_list.append(default_numeric_value)
            elif var_type[var] == "TEXT":
                payload_value_list.append(default_text_value)
            else:
                raise ValueError

    # Put sample data in your payload
    sample_payload = ",".join(payload_value_list)
    with open("payload.csv", "w") as f:
        f.write(sample_payload)

    print("cwd =", os.getcwd())

    print("create payload tar.gz")

    with tarfile.open(os.path.join(payload_dir, "payload.tar.gz"), "w:gz") as tar:
        tar.add("payload.csv")
