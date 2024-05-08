"""
Created a set of heuristics to recover causal splice junctions in the muscle truth set.
"""

import numpy as np
import pandas as pd
import hail as hl
import gzip

CHROM_SET = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
DESCRIPTION_COLS = ["motif", "uniquely_mapped", "multi_mapped",
                    "maximum_spliced_alignment_overhang", "annotated_junction",
                    "num_samples_with_this_junction", "num_samples_total"]

DISEASE_GENES = {"PYROXD1": "chr12:21437615-21471250:+",
                 "NEB": "chr2:151485336-151734487:-",
                 "RYR1": "chr19:38433691-38595273:+",
                 "TTN": "chr2:178525989-178830802:-",
                 "COL6A3": "chr2:237324003-237414328:-",
                 "POMGNT1": "chr1:46188683-46220305:-",
                 "DMD": "chrX:31097677-33339609:-",
                 "COL6A1": "chr21:45981770-46005050:+",
                 "CAPN3": "chr15:42359498-42412949:+",
                 "LARGE1": "chr22:33162226-33922841:-",
                 "SRPK3": "chrX:153776412-153785732:+",
                 "MTM1": "chrX:150568417-150673143:+",
                 "LAMA2": "chr6:128883138-129516566:+",
                 "COL6A2": "chr21:46098112-46132848:+"}

TRUTH_SET = {"41M_MW_M1": "chr12:21440448-21449562",
             "153BR_JB_M1": "chr2:151672680-151675286",
             "163BV_JE_M1": "chr2:151680828-151684777",
             "210DB_BW_M1": "chr19:38440864-38443557",
             "T1244": "chr2:178622008-178624464",
             "361AL_CC_M1": "chr2:237358583-237359205",
             "149BP_AB_M1": "chr2:151498552-151499297",
             "UC84-1RNA": "chr2:151531896-151534214",
             "247DT_SH_M1": "chr2:151687733-151688371",
             "373HQ_BTG_M1": "chr2:178580609-178581905",
             "B11-48-1M": "chr1:46189357-46189457",
             "TOP_MAAC031_F_Muscle1": "chrX:31590401-31595570",
             "26I_SK_M1": "chr2:151531896-151533382",
             "CLA_180CJ_DP_2": "chr19:38467812-38468965",
             "UC316-1M": "chr21:45989778-45989893",
             "UC393-1M": "chr21:45989778-45989893",
             "49O_NM_M1": "chrX:32256704-32287528",
             "MBEL028_002_1": "chr15:42399885-42401640",
             "251DW_SD_M1": "chrX:31729748-31819974",
             "126BG_CB_M1": "chr22:33337801-33919994",
             "MBRU030_2": "chrX:153781828-153782120",
             "B14-78-1-U": "chr6:129453131-129454154",
             "205E_BD_M1": "chr21:46126237-46131953"}

BATCH_2023 = {"RGP_2058_3_M1", "BON_B22_12_1_R1"}
BATCH_2022 = {"BON_B16-59_1"}

""" future use
"251DW_SD_M1": "chrX:31729748-31819974",
"253DY_HA_M1": "chrX:32518008",  # outside of all splice junctions?
"126BG_CB_M1": "chr22:33337801-33919994",
"RGP_2058_3_M1": "chr21:45989778-45989874",  # does not have num_samples_with_this_junction
"BON_B22_12_1_R1": "chr19:38505946-285063781",  # does not have num_samples_with_this_junction
"MBRU030_2": "chrX:153781828-153782120", 
"LIA_MAS02_2": "chrX:31118218-33340460", # causal sj not exact
"BEG_1025-1_T999": "chrX:150598686",  # 150598686-150614588 (1 umapped) or 150598686-150638942 (0 umapped)
"B14-78-1-U": "chr6:129453131-129453132",
"205E_BD_M1": "chr21:46126238-46131953",
"BON_B16-59_1": "chr21:45989778-45990257", # does not have num_samples_with_this_junction
"""


def convert_to_int(x):
    try:
        int(x)
        return int(x)
    except ValueError:
        return x


def filter_to_disease_genes(dat):
    filtered_bed_dat = pd.DataFrame([])
    for cur_disease_gene in DISEASE_GENES:
        cur_dat = dat[
            (dat["chr"] == DISEASE_GENES[cur_disease_gene].split(":")[0]) &
            (dat["start"] >= int(
                DISEASE_GENES[cur_disease_gene].split(":")[1].split("-")[0])) &
            (dat["end"] <= int(
                DISEASE_GENES[cur_disease_gene].split(":")[1].split("-")[1]))]
        # (dat["strand"] == DISEASE_GENES[cur_disease_gene].split(":")[2])]

        if cur_dat.shape[0] == 0:
            print(f"No junctions found for {cur_disease_gene}.")
        filtered_bed_dat = pd.concat([filtered_bed_dat, cur_dat])

    return filtered_bed_dat


def is_present_causal_sj(filtered_bed_dat, causal_sj):
    causal_sj_row = filtered_bed_dat[
        (filtered_bed_dat["chr"] == causal_sj.split(":")[0]) &
        (filtered_bed_dat["start"] == int(causal_sj.split(":")[1].split("-")[0])) &
        (filtered_bed_dat["end"] == int(causal_sj.split(":")[1].split("-")[
                                            1]))]
    print(causal_sj_row)
    return causal_sj_row.shape[0] > 0


def read_junctions_bed(sample_id, junctions_bed_path, use_cols=None):
    bed_dat = pd.read_csv(junctions_bed_path, sep="\t", header=None)
    bed_dat.columns = ["chr", "start", "end", "description", "uniquely_mapped",
                       "strand"]
    bed_dat = bed_dat[bed_dat["chr"].isin(CHROM_SET)]
    bed_dat = bed_dat.drop(columns=["uniquely_mapped"])

    bed_dat[DESCRIPTION_COLS] = bed_dat[
        "description"].str.split(";", expand=True)
    bed_dat = bed_dat.drop(columns=["description"])
    for cur_col in DESCRIPTION_COLS:
        bed_dat[cur_col] = bed_dat[cur_col].str.split("=").str[1]
        bed_dat[cur_col] = bed_dat[cur_col].apply(convert_to_int)

    if use_cols is not None:
        bed_dat = bed_dat[use_cols]

    print("-----------------------------------")
    print(bed_dat.shape)
    # Decision rules to filter out splice junctions.
    # 1. Keep only splice junctions that have at least 1 uniquely mapped reads.
    print(np.mean(bed_dat["uniquely_mapped"].astype(int)),
          np.median(bed_dat["uniquely_mapped"].astype(int)))
    bed_dat = bed_dat[bed_dat["uniquely_mapped"] > 0]
    print(bed_dat.shape)

    # 2. Filter to known disease genes.
    filtered_bed_dat = filter_to_disease_genes(bed_dat)
    print(filtered_bed_dat.shape)
    casual_sj = TRUTH_SET[sample_id]
    is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)

    # 3. Filter by number of samples with this junction.
    # filtered_bed_dat = filtered_bed_dat[
    #     filtered_bed_dat["num_samples_with_this_junction"] < 10]
    # print(filtered_bed_dat.shape)

    # 4. Compare with GTEx data.

    # Check if the causal splice junction is in the filtered data.
    casual_sj = TRUTH_SET[sample_id]
    is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)
    print(f"{sample_id}: {casual_sj}")
    print(is_present)
    print(f"The total number of splice junctions after filtering is"
          f" {filtered_bed_dat.shape[0]}.")

    # pd.set_option('display.max_columns', None)
    # print(filtered_bed_dat)
    return is_present


def main():
    captured_samples = []
    for cur_sample in TRUTH_SET:
        if cur_sample in BATCH_2023:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_2023_01/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
        elif cur_sample in BATCH_2022:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_2022_01/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
        else:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_0/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"

        use_cols = ["chr", "start", "end", "strand", "uniquely_mapped",
                    "maximum_spliced_alignment_overhang",
                    "num_samples_with_this_junction",
                    "num_samples_total"]
        is_present = read_junctions_bed(cur_sample, junctions_bed_path, use_cols)
        if is_present:
            captured_samples.append(cur_sample)
    print(f"In total, {len(captured_samples)} out of {len(TRUTH_SET)} causal splice "
          f"junctions are "
          f"captured in "
          f"the truth set.")
    print(captured_samples)


if __name__ == "__main__":
    main()
