"""
Created a set of heuristics to recover causal splice junctions in the muscle truth set.
"""
import os

import numpy as np
import pandas as pd
import hail as hl
import gzip
import pyBigWig
from scipy import stats

CHROM_SET = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
DESCRIPTION_COLS = ["motif", "uniquely_mapped", "multi_mapped",
                    "maximum_spliced_alignment_overhang", "annotated_junction",
                    "num_samples_with_this_junction", "num_samples_total"]
USE_COLS = ["chr", "start", "end", "strand", "uniquely_mapped",
            "maximum_spliced_alignment_overhang",
            "num_samples_with_this_junction",
            "num_samples_total"]

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

GTEx = "gs://tgg-viewer/ref/GRCh38/gtex_v8/GTEX_muscle.803_samples.junctions.bed.gz"
GTEx_NORMALIZED_BED = "gs://tgg-viewer/ref/GRCh38/gtex_v8/GTEX_muscle.803_samples.normalized.junctions.bed.gz"
GTEX_NORMALIZED_BIGWIG = "gs://tgg-viewer/ref/GRCh38/gtex_v8/GTEX_muscle.803_samples.normalized.bigWig"

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


def filter_to_disease_genes(dat, genes_dict):
    filtered_bed_dat = pd.DataFrame([])
    for cur_disease_gene in genes_dict:
        cur_dat = dat[
            (dat["chr"] == genes_dict[cur_disease_gene].split(":")[0]) &
            (dat["start"] >= int(
                genes_dict[cur_disease_gene].split(":")[1].split("-")[0])) &
            (dat["end"] <= int(
                genes_dict[cur_disease_gene].split(":")[1].split("-")[1]))].copy()
        cur_dat["gene"] = cur_disease_gene
        # (dat["strand"] == genes_dict[cur_disease_gene].split(":")[2])

        if cur_dat.shape[0] == 0:
            print(f"No junctions found for {cur_disease_gene}.")
        filtered_bed_dat = pd.concat([filtered_bed_dat, cur_dat])

    return filtered_bed_dat


# TODO: Change the & between start and end to |.
# How to define if a splice junction is present in the filtered data based on the
# 1kb-10kb window?
def is_present_causal_sj(filtered_bed_dat, causal_sj):
    causal_sj_row = filtered_bed_dat[
        (filtered_bed_dat["chr"] == causal_sj.split(":")[0]) &
        (filtered_bed_dat["start"] == int(causal_sj.split(":")[1].split("-")[0])) &
        (filtered_bed_dat["end"] == int(causal_sj.split(":")[1].split("-")[
                                            1]))]
    print(causal_sj_row)
    return causal_sj_row.shape[0] > 0


def filter_by_uniquely_mapped_reads(dat, threshold):
    # print(np.mean(dat["uniquely_mapped"].astype(int)),
    #       np.median(dat["uniquely_mapped"].astype(int)))
    dat = dat[dat["uniquely_mapped"] >= threshold]
    return dat


def get_sj_only_in_first_dat(dat1, dat2):
    merged_dat = pd.merge(dat1, dat2, how="left", on=["chr", "start", "end"],
                          suffixes=('', '_y'),
                          indicator=True).query('_merge == "left_only"').drop(
        columns=['_merge'])
    merged_dat = merged_dat.drop(
        columns=[col for col in merged_dat.columns if col.endswith("_y")])
    return merged_dat


def get_sj_different_between_dats(dat1, dat2, lr_sj_threshold, lr_coverage_threshold):
    pd.set_option('display.max_columns', None)
    merged_dat = pd.merge(dat1, dat2, how='inner', on=["chr", "start", "end"],
                          suffixes=('', '_y'))
    pseudocount = 1e-12
    sj_ratio = (merged_dat["normalized_by_sj"] + pseudocount) / (merged_dat[
                                                                     "normalized_by_sj_y"] + pseudocount)

    coverage_ratio = (merged_dat["normalized_by_coverage"] + pseudocount) / (merged_dat[
                                                                                 "normalized_by_coverage_y"] + pseudocount)
    # for i in range(coverage_ratio.shape[0]):
    #     if coverage_ratio[i] < 0:
    #         print(merged_dat.iloc[i, :])

    merged_dat["lr_sj"] = abs(np.log2(sj_ratio))
    merged_dat["lr_coverage"] = abs(np.log2(coverage_ratio))
    # # Test
    # pd.set_option('display.max_columns', None)
    # cur_sample = "UC84-1RNA"
    # causal_sj = TRUTH_SET[cur_sample]
    # is_present_causal_sj(merged_dat, causal_sj)
    # #

    pd.set_option('display.max_columns', None)
    # print(merged_dat)
    # is_present_causal_sj(dat1, TRUTH_SET["B14-78-1-U"])
    # print("-----------------------------------")
    # is_present_causal_sj(dat2, TRUTH_SET["B14-78-1-U"])
    # print("-----------------------------------")
    is_present_causal_sj(merged_dat, TRUTH_SET["CLA_180CJ_DP_2"])
    print("-----------------------------------")

    # merged_dat = merged_dat[(merged_dat["lr_sj"] > lr_sj_threshold)]
    merged_dat = merged_dat[(merged_dat["lr_sj"] > lr_sj_threshold) | (
            merged_dat["lr_coverage"] > lr_coverage_threshold)]
    # merged_dat = merged_dat[merged_dat["lr_coverage"] > lr_coverage_threshold]
    merged_dat = merged_dat.drop(
        columns=[col for col in merged_dat.columns if col.endswith("_y")])
    # is_present_causal_sj(merged_dat, TRUTH_SET["CLA_180CJ_DP_2"])
    merged_dat = merged_dat.drop(columns=["lr_sj", "lr_coverage"])

    return merged_dat


def compare_with_gtex(gtex_normalized_dat, dat, lr_sj_threshold, lr_coverage_threshold):
    # in_rd_not_in_gtex_dat = get_sj_only_in_first_dat(dat, gtex_dat)
    in_rd_not_in_gtex_normalized_dat = get_sj_only_in_first_dat(dat,
                                                                gtex_normalized_dat)
    # in_gtex_not_in_rd_dat = get_sj_only_in_first_dat(gtex_dat, dat)
    in_gtex_normalized_not_in_rd_dat = get_sj_only_in_first_dat(gtex_normalized_dat,
                                                                dat)
    different_between_rd_and_gtex = get_sj_different_between_dats(dat,
                                                                  gtex_normalized_dat,
                                                                  lr_sj_threshold=2,
                                                                  lr_coverage_threshold=2)

    res = pd.concat([in_rd_not_in_gtex_normalized_dat,
                     # in_gtex_normalized_not_in_rd_dat,
                     different_between_rd_and_gtex])
    return res
    # pass
    # diff_dat = pd.concat([in_rd_not_in_gtex_normalized_dat,
    #                       in_gtex_normalized_not_in_rd_dat],
    #                      axis=0, ignore_index=True)
    # return diff_dat


def read_junctions_bed(junctions_bed_path, use_cols=None):
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

    return bed_dat


def read_coverage_bigwig(bigwig_path):
    local_path = os.path.basename(bigwig_path)
    if not os.path.exists(local_path):
        os.system(f"gcloud storage cp {bigwig_path} {local_path}")

    bw = pyBigWig.open(local_path)

    return bw


def merge_overlapping_segments(dat):  # TODO: check the algorithm
    dat = dat.sort_values(by=["start", "end"])
    segments = []
    last_start = None
    last_end = None
    chr = dat["chr"].iloc[0]
    for index, row in dat.iterrows():
        start = row["start"]
        end = row["end"]

        if last_start is None:
            last_start = start
            last_end = end
        else:
            if start <= last_end:
                last_end = max(last_end, end)
            else:
                segments.append((chr, last_start, last_end))
                last_start = start
                last_end = end
    segments.append((chr, last_start, last_end))
    return segments


def normalize_reads_by_gene(dat, coverage_bw):
    """
    Normalize the number of reads by dividing the reads of a splice junction by the
    total number of reads of all splice junctions within the gene. Also, normalize the
    reads by the coverage of the gene.
    """

    # Normalize by total reads of splice junctions within the gene.
    gene_total_sj_reads = dat.groupby("gene")["uniquely_mapped"].sum().reset_index()
    gene_total_sj_reads.columns = ["gene", "total_reads"]
    gene_total_sj_reads_dict = {}
    for i in range(gene_total_sj_reads.shape[0]):
        gene_total_sj_reads_dict[gene_total_sj_reads.iloc[i, 0]] = \
            gene_total_sj_reads.iloc[i, 1]

    # print(gene_total_sj_reads_dict)
    dat["normalized_by_sj"] = dat["uniquely_mapped"] / dat["gene"].map(
        gene_total_sj_reads_dict)

    # Normalize by the coverage of the gene.
    iqs_intron_coverage_dict = {}
    mean_gene_coverage_dict = {}
    for gene in DISEASE_GENES:
        gene_chr = DISEASE_GENES[gene].split(":")[0]
        gene_start = int(DISEASE_GENES[gene].split(":")[1].split("-")[0])
        gene_end = int(DISEASE_GENES[gene].split(":")[1].split("-")[1])

        # total_exon_coverage, num_of_bases, all_bases = get_exon_coverage(coverage_bw,
        #                                                                  gene_chr,
        #                                                                  gene_start,
        #                                                                  gene_end,
        #                                                                  gene)
        # mean_gene_coverage_dict[gene] = total_exon_coverage / num_of_bases


    # print(mean_gene_coverage_dict)
    dat["sj_coverage"] = dat.apply(lambda row: get_intron_coverage(coverage_bw,
                                                                   row["chr"],
                                                                   row["start"],
                                                                   row["end"]),
                                   axis=1)

    # pd.set_option('display.max_columns', None)
    # is_present_causal_sj(dat, TRUTH_SET["CLA_180CJ_DP_2"])
    # print("-----------------------------------")
    dat["normalized_by_coverage"] = dat["sj_coverage"] - dat["gene"].map(
        mean_gene_coverage_dict)
    print(mean_gene_coverage_dict["RYR1"])

    return dat


def get_exon_coverage(bw, chr, start, end, gene_name=None):
    if gene_name is None:
        cur_mane_exons = mane_exons_in_disease_genes[
            (mane_exons_in_disease_genes["chr"] == chr) &
            (mane_exons_in_disease_genes["start"] >= start) &
            (mane_exons_in_disease_genes["end"] <= end)]
    else:
        cur_mane_exons = mane_exons_in_disease_genes[
            mane_exons_in_disease_genes["gene"] == gene_name]

    total_exon_coverage = 0
    num_of_bases = 0
    all_bases = []
    for i in range(cur_mane_exons.shape[0]):
        chr = cur_mane_exons.iloc[i, 0]
        start = cur_mane_exons.iloc[i, 1]
        end = cur_mane_exons.iloc[i, 2]
        num_of_bases += end - start

        coverage = list(bw.values(chr, start, end))
        total_exon_coverage += np.nansum(coverage)
        all_bases += coverage

    return total_exon_coverage, num_of_bases, all_bases


def get_intron_coverage(bw, chr, start, end, gene_name=None):
    exon_coverage = get_exon_coverage(bw, chr, start, end, gene_name)[0]
    total_coverage = np.nansum(bw.values(chr, start, end))
    return total_coverage - exon_coverage


def apply_filters(bed_dat, sample_id, gtex_normalized_dat, bigwig_path):
    print("-----------------------------------")
    print(bed_dat.shape)
    # Decision rules to filter out splice junctions.
    # 1. Keep only splice junctions that have at least 1 uniquely mapped reads.
    bed_dat = filter_by_uniquely_mapped_reads(bed_dat, 1)
    print(bed_dat.shape)

    # 2. Filter to known disease genes.
    filtered_bed_dat = filter_to_disease_genes(bed_dat, DISEASE_GENES)
    filtered_gtex_normalized_dat = filter_to_disease_genes(gtex_normalized_dat,
                                                           DISEASE_GENES)
    print(filtered_bed_dat.shape)
    # casual_sj = TRUTH_SET[sample_id]
    # is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)

    # 3. Compare reads and coverage with GTEx data.
    # Get the coverage file.
    bw = read_coverage_bigwig(bigwig_path)
    gtex_bw = read_coverage_bigwig(GTEX_NORMALIZED_BIGWIG)

    normalized_filtered_bed_dat = normalize_reads_by_gene(filtered_bed_dat, bw)
    normalized_gtex_normalized_dat = normalize_reads_by_gene(
        filtered_gtex_normalized_dat, gtex_bw)
    filtered_bed_dat = compare_with_gtex(normalized_gtex_normalized_dat,
                                         normalized_filtered_bed_dat,
                                         lr_sj_threshold=2,
                                         lr_coverage_threshold=2)
    print(filtered_bed_dat.shape)
    grouped_gene = filtered_bed_dat.groupby("gene").size()
    print(grouped_gene)

    # 4. Compare coverage with GTEx data.

    # Close the BigWig file
    bw.close()
    # os.system(f"rm *.bigWig")

    # 5. Filter by maximum spliced alignment overhang.
    # filtered_bed_dat = filtered_bed_dat[filtered_bed_dat[
    #                                         "maximum_spliced_alignment_overhang"] > 2]
    # print(filtered_bed_dat.shape)

    # 5. Filter by number of samples with this junction.
    # filtered_bed_dat = filtered_bed_dat[
    #     filtered_bed_dat["num_samples_with_this_junction"] < 10]
    # print(filtered_bed_dat.shape)

    # pd.set_option('display.max_columns', None)
    # print(filtered_bed_dat)
    return filtered_bed_dat


def main():
    captured_samples = []
    number_of_candidates = []
    # gtex_dat = read_junctions_bed(GTEx, USE_COLS)
    # gtex_dat = filter_by_uniquely_mapped_reads(gtex_dat, 1)

    gtex_normalized_dat = read_junctions_bed(GTEx_NORMALIZED_BED, USE_COLS)
    gtex_normalized_dat = filter_by_uniquely_mapped_reads(gtex_normalized_dat, 1)

    for cur_sample in TRUTH_SET:
        if cur_sample != "CLA_180CJ_DP_2":
            continue
        if cur_sample in BATCH_2023:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_2023_01/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
        elif cur_sample in BATCH_2022:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_2022_01/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
        else:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_0/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"

        bigwig_path = f"gs://tgg-rnaseq/batch_0/bigWig/{cur_sample}.bigWig"

        bed_dat = read_junctions_bed(junctions_bed_path, USE_COLS)
        filtered_bed_dat = apply_filters(bed_dat, cur_sample,
                                         gtex_normalized_dat,
                                         bigwig_path)

        number_of_candidates.append(filtered_bed_dat.shape[0])
        # Check if the causal splice junction is in the filtered data.
        casual_sj = TRUTH_SET[cur_sample]
        is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)
        print(f"{cur_sample}: {casual_sj}")
        print(is_present)
        print(f"The total number of splice junctions after filtering is"
              f" {filtered_bed_dat.shape[0]}.")

        if is_present:
            captured_samples.append(cur_sample)

    print(f"In total, {len(captured_samples)} out of {len(TRUTH_SET)} causal splice "
          f"junctions are "
          f"captured in "
          f"the truth set.")
    print(captured_samples)
    print(np.mean(number_of_candidates), np.median(number_of_candidates))


if __name__ == "__main__":
    # bigwig_path = "gs://tgg-rnaseq/batch_0/bigWig/210DB_BW_M1.bigWig"
    # bw = read_coverage_bigwig(bigwig_path)
    # values = bw.values("chr19", 1, 3)
    # print(values)
    mane_exons = pd.read_csv("MANE_exon_1based_start.tsv", sep="\t")
    mane_exons_in_disease_genes = filter_to_disease_genes(mane_exons, DISEASE_GENES)
    print(mane_exons_in_disease_genes)
    main()
#     dat1 = pd.DataFrame([["chr1", 1, 2, 5, "A"], ["chr1", 3, 4, 5, "A"], ["chr1",
# 3, 10, 5, "B"]])
#     dat1.columns = ["chr", "start", "end", "uniquely_mapped", "gene"]
#     dat1_normalized = normalize_reads_by_gene(dat1)
#     print(dat1_normalized)
#
#     dat2 = pd.DataFrame([["chr1", 1, 2, 2, "A"], ["chr1", 3, 4, 10, "A"], ["chr1",
#                                                                           3, 10, 5,
#                                                                           "B"]])
#     dat2.columns = ["chr", "start", "end", "uniquely_mapped", "gene"]
#     dat2_normalized = normalize_reads_by_gene(dat2)
#     print(dat2_normalized)
#     get_sj_different_between_dats(dat1_normalized, dat2_normalized, 0.1)

# dat2 = pd.DataFrame([["chr1", 1, 2, 3], ["chr1", 3, 5, 3], ["chr2", 5, 7, 3]])
# dat2.columns = ["chr", "start", "end", "count"]
# print(get_sj_only_in_first_dat(dat1, dat2))
# gtex_dat = read_junctions_bed(GTEx_normalized, USE_COLS)
# gtex_dat = filter_by_uniquely_mapped_reads(gtex_dat, 1)
#
# junctions_bed_path = "gs://tgg-rnaseq/batch_0/junctions_bed_for_igv_js/41M_MW_M1.junctions.bed.gz"
# bed_dat = read_junctions_bed(junctions_bed_path, USE_COLS)
#
# # new_dat = get_sj_only_in_first_dat(gtex_dat, bed_dat)
# new_dat = compare_with_gtex(gtex_dat, bed_dat)
#
# pd.set_option('display.max_columns', None)
# print(new_dat)
# print(new_dat.columns)
# print(gtex_dat.shape)
# print(bed_dat.shape)
