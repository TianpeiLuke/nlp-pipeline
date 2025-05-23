import os
import tarfile
import pickle as pkl



def create_payload():
    payload = {}
    for field in full_field_list:
        if field in tab_field_list:
            payload[field] = default_numeric_value
        elif field in cat_field_list:
            payload[field] = default_text_value
        # Ignore label and id fields in the payload
        elif field not in [label_name, id_name]:
            # For any fields not explicitly categorized, default to text
            payload[field] = default_text_value
    return payload