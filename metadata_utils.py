import os
import pandas as pd
import yaml
from pyairtable import Api

_GSPREAD_CLIENT = None

RNA_SEQ_BASE_ID = "app034nYTJwpo9xxw"
"""
The base ID of the `data_paths` RNA-seq metadata Airtable.
"""
DATA_PATHS_TABLE_ID = "tblmT2O7VQasTKSDx"
"""
The table ID of the `data_paths` RNA-seq metadata Airtable.
"""
DATA_PATHS_VIEW_ID = "viwUzuAxagca7zF0l"
"""
The view ID of the `data_paths` RNA-seq metadata Airtable.
"""
METADATA_TABLE_ID = "tblcluIOjz23rZDya"
"""
The table ID of the `sample_metadata` RNA-seq metadata Airtable.
"""
METADATA_VIEW_ID = "viwTjCsk3SccsOJIz"
"""
The view ID of the `sample_metadata` RNA-seq metadata Airtable.
"""
DOWNSTREAM_ANALYSIS_TABLE_ID = "tblyJvv0ozZjXIutA"
"""
The table ID of the `downstream_analysis` RNA-seq metadata Airtable.
"""
DOWNSTREAM_ANALYSIS_VIEW_ID = "viwkCT48ws9yDTDmf"
"""
The view ID of the `downstream_analysis` RNA-seq metadata Airtable.
"""

GCLOUD_CREDENTIALS = "gs://jialan-cred"
GCLOUD_USR_ACCOUNT = "majialan@broadinstitute.org"
GCLOUD_PROJECT = "tgg-rare-disease"

# Parse YAML config file containing credentials: service account credentials and
# Airtable token.
config_file_path = os.path.expanduser("~/.tgg_rnaseq_pipelines")
if not os.path.isfile(config_file_path):
    raise Exception(f"Config file not found: {config_file_path}")

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Check if credentials exist.
credentials = ["service-account", "airtable-token"]
for credential in credentials:
    if credential not in config:
        raise Exception(
            f'Config file {config_file_path} doesn\'t contain a "{credential}" key'
        )

### no use of the path since we do not use google spreadsheet?
SERVICE_ACCOUNT_CREDENTIALS_JSON_PATH = os.path.expanduser(
    str(config["service-account"])
)
# Get the Airtable token to access Airtable through Airtable APIs.
AIRTABLE_TOKEN = str(config["airtable-token"])


def normalize_rnaseq_sample_id(
        sample_id: str,
):
    """
    Normalize the format of a sample ID: substitute "." with "-" and get rid of
    trailing spaces.

    :param sample_id: the sample ID to normalize.
    :return: normalized sample ID as str.
    """
    sample_id = sample_id.strip()
    sample_id = sample_id.replace(".", "-")
    return sample_id


def switch_account_for_gtex(batch, batch_job):
    switch_gcloud_auth_to_user_account(batch,
                                       batch_job,
                                       gcloud_credentials_path=GCLOUD_CREDENTIALS,
                                       gcloud_user_account=GCLOUD_USR_ACCOUNT,
                                       gcloud_project=GCLOUD_PROJECT)


def gcloud_auth_activate_service_account(batch_job):
    """
    Utility method to active gcloud auth using the Hail Batch-provided service account
    """
    batch_job.command(
        f"gcloud auth activate-service-account --key-file /gsa-key/key.json")


def switch_gcloud_auth_to_user_account(batch,
                                       batch_job,
                                       gcloud_credentials_path,
                                       gcloud_user_account,
                                       gcloud_project):
    if not gcloud_credentials_path:
        raise ValueError("gcloud_credentials_path not specified.")

    if not gcloud_user_account:
        raise ValueError("gcloud_user_account not specified.")

    if not gcloud_project:
        raise ValueError("gcloud_project not specified.")

    gcloud_auth_activate_service_account(batch_job)
    config_path = os.path.join(gcloud_credentials_path, '.config')
    localized_config_path = batch.read_input(config_path)
    batch_job.command(f"ln -s {localized_config_path} /tmp/")

    batch_job.command(f"cp -r /tmp/.config/* ~/.config/")
    batch_job.command(f"gcloud config set account {gcloud_user_account}")
    batch_job.command(f"gcloud config set project {gcloud_project}")
    batch_job.command(
        f"export GOOGLE_APPLICATION_CREDENTIALS=$(find ~/.config/ -name 'adc.json')")


def get_table(
        base_id: str,
        table_id: str,
        view_id: str,
):
    """
    Get the data table from Airtable.

    :param base_id: the base ID of the data table in Airtable.
    :param table_id: the table ID of the data table in Airtable.
    :param view_id: the view ID of the data table in Airtable.
    :return: a data table as Python list.
    """
    api = Api(AIRTABLE_TOKEN)
    all_tables = api.table(base_id, table_id)
    cur_table = all_tables.all(view=view_id)
    return cur_table


def read_from_airtable(
        base_id: str,
        table_id: str,
        view_id: str,
):
    """
    Get the data table as a pandas DataFrame from Airtable.

    :param base_id: the base ID of the data table in Airtable.
    :param table_id: the table ID of the data table in Airtable.
    :param view_id: the view ID of the data table in Airtable.
    :return: a data table as pandas DataFrame.
    """
    cur_table = get_table(base_id, table_id, view_id)
    # Only keep information in `fields`.
    for i in range(len(cur_table)):
        record = cur_table[i]
        cur_table[i] = record["fields"]

    df = pd.DataFrame(cur_table)
    return df


def map_airtable_and_sample_id(
        base_id: str,
        table_id: str,
        view_id: str,
):
    """
    Create a Python dictionary that maps RNA-seq sample IDs to their record IDs in
    the Airtable.

    :param base_id: the base ID of the data table in Airtable.
    :param table_id: the table ID of the data table in Airtable.
    :param view_id: the view ID of the data table in Airtable.
    :return: a Python dict.
    """
    cur_table = get_table(base_id, table_id, view_id)
    map_dict = {}
    for i in range(len(cur_table)):
        record = cur_table[i]
        record_id = record["id"]
        sample_id = record["fields"]["sample_id"]
        map_dict[sample_id] = record_id
    return map_dict


def get_missing_records(
        base_id: str,
        table_id: str,
        view_id: str,
        column: str,
):
    """
    Get the records that have an empty value in the specified column in Airtable.

    :param base_id: the base ID of the data table in Airtable.
    :param table_id: the table ID of the data table in Airtable.
    :param view_id: the view ID of the data table in Airtable.
    :param column: the name of the column in Airtable.
    :return: a pandas DataFrame containing records with an empty value in the column;
    a Python list containing record IDs of these records.
    """
    cur_table = get_table(base_id, table_id, view_id)
    records_with_na_in_column = []
    ids_with_na_in_column = []
    for i in range(len(cur_table)):
        record = cur_table[i]
        if column and column not in record["fields"].keys():
            records_with_na_in_column.append(record["fields"])
            ids_with_na_in_column.append(record["id"])

    records_with_na_in_column_df = pd.DataFrame(records_with_na_in_column)
    return records_with_na_in_column_df, ids_with_na_in_column


def write_to_airtable(
        base_id: str,
        table_id: str,
        new_records: list,
        to_replace: bool = False,
        fields_to_match: list = ["sample_id"],
        re_sequenced: bool = False,
):
    """
    Update current records or insert new records to the Airtable.

    :param base_id: the base ID of the data table in Airtable.
    :param table_id: the base ID of the data table in Airtable.
    :param new_records: the records to update/insert.
    :param to_replace: if True, replace the currently existing records; else, skip.
    :param fields_to_match: the field to locate the record in Airtable.
    :param re_sequenced: if True, insert the record instead of update the record.
    """
    api = Api(AIRTABLE_TOKEN)
    all_tables = api.table(base_id, table_id)
    if re_sequenced:
        all_tables.batch_create(new_records)
    else:
        all_tables.batch_upsert(new_records,
                                key_fields=fields_to_match,
                                replace=to_replace)


def inner_join_two_tables_into_df(
        base_id1: str,
        table_id1: str,
        view_id1: str,
        base_id2: str,
        table_id2: str,
        view_id2: str,
):
    """
    Inner join two Airtable data tables and remove duplicated columns.

    :param base_id1: the base ID of the data table in Airtable.
    :param table_id1: the table ID of data table 1 in Airtable.
    :param view_id1: the view ID of data table 1 in Airtable.
    :param base_id2: the base ID of data table 2 in Airtable.
    :param table_id2: the table ID of data table 2 in Airtable.
    :param view_id2: the view ID of data table 2 in Airtable.
    :return: the joined data table as pandas DataFrame.
    """
    df1 = read_from_airtable(base_id1, table_id1, view_id1)
    df2 = read_from_airtable(base_id2, table_id2, view_id2)
    df_res = pd.merge(
        df1, df2, how="inner", left_index=True, right_index=True, suffixes=("", "_drop")
    )
    df_res.drop([col for col in df_res.columns if "drop" in col], axis=1, inplace=True)
    return df_res
