import pandas as pd
import numpy as np
import os
import ast
import json
import re
from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import (
    read_from_airtable, RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
from set_heuristics import TRUTH_SET
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import hailtop.batch as hb
import argparse

REGION = ["us-central1"]
DOCKER_IMAGE = "gcr.io/cmg-analysis/vcf-annotation:latest"

def rename_vcf_file(vcf_path):
    pattern = r"\.vcf.*gz$"
    new_vcf_path = re.sub(pattern, ".annotated.vcf.bgz", vcf_path)
    return new_vcf_path


def run_job(cmd):
    print(f"Starting: {cmd}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Finished: {cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--billing-project", type=str, help="Project to bill under.",
                        default="tgg-rare-disease")
    parser.add_argument("--requester-pays-project", type=str,
                        help="Requester pays project to bill under.",
                        default="cmg-analysis")
    parser.add_argument("--file-dir", type=str,
                        help="The directory to store results table.",
                        default="gs://jialan-storage")
    args = parser.parse_args()

    backend = hb.ServiceBackend(billing_project=args.billing_project,
                                remote_tmpdir=args.file_dir,
                                regions=REGION)

    batch_name = "vcf-annotation"
    batch = hb.Batch(backend=backend,
                     name=batch_name,
                     requester_pays_project=args.requester_pays_project,
                     default_image=DOCKER_IMAGE,
                     # default_cpu=DEFAULT_CPU,
                     # default_memory=DEFAULT_MEMORY,
                     # default_storage=DEFAULT_STORAGE,
                     )

    # Read data from Airtable.
    rdg_dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID,
                                 DATA_PATHS_VIEW_ID)
    rdg_dat = rdg_dat[~(rdg_dat["exclude"] == "yes")]
    rdg_dat = rdg_dat[rdg_dat["imputed_tissue"] == "muscle"]
    rdg_dat = rdg_dat[rdg_dat["RQS"] >= 6]
    rdg_dat = rdg_dat[rdg_dat["wgs_single_sample_vcf"].notna() | rdg_dat[
        "wes_single_sample_vcf"].notna()]
    # rdg_dat = rdg_dat[rdg_dat["sample_id"].isin(TRUTH_SET.keys())]

    sample_ids = list(rdg_dat["sample_id"])
    wgs_vcf = list(rdg_dat["wgs_single_sample_vcf"])
    wes_vcf = list(rdg_dat["wes_single_sample_vcf"])
    sample_vcf_dict = {}
    cmds = []
    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        wgs_vcf_path = wgs_vcf[i]
        wes_vcf_path = wes_vcf[i]

        # if sample_id != "149BP_AB_M1" and sample_id != "BON_B20-16_1":
        #     continue
        print(sample_id)
        cur_job = batch.new_job(name=f"{sample_id}_add_dna_variant")
        if not pd.isna(wgs_vcf_path):
            output_vcf = rename_vcf_file(wgs_vcf_path)
            # cmd = f"hailctl dataproc submit jm add_dna_variant.py --input-vcf " \
            #       f"{wgs_vcf_path} --output-vcf {output_vcf} --sample-id {sample_id}".split(" ")
            # cmds.append(cmd)
            cur_job.command(f"python3 add_dna_variant.py --input-vcf {wgs_vcf_path} "
                           f"--output-vcf {output_vcf} --sample-id {sample_id}")
        else:
            output_vcf = rename_vcf_file(wes_vcf_path)
            # cmd = f"hailctl dataproc submit jm add_dna_variant.py --input-vcf " \
            #       f"{wes_vcf_path} --output-vcf {output_vcf} --sample-id {sample_id}".split(" ")
            # cmds.append(cmd)
            cur_job.command(f"python3 add_dna_variant.py --input-vcf {wes_vcf_path} "
                           f"--output-vcf {output_vcf} --sample-id {sample_id}")

        sample_vcf_dict[sample_id] = output_vcf
        break

    # with ThreadPoolExecutor(max_workers=50) as executor:
    #     futures = [executor.submit(run_job, cmd) for cmd in cmds]

    print(sample_vcf_dict)

    # Dump the sample_vcf_dict to a json file.
    with open("sample_vcf_dict.json", "w") as f:
        json.dump(sample_vcf_dict, f, indent=4)
    # Rerun gs://tgg-rnaseq/batch_0/grch38_vcfs/149BP_AB_M1.vcf.bgz
    batch.run()