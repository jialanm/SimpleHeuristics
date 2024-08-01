"""
Created a set of heuristics to recover causal splice junctions in the muscle truth set.
"""
import os

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pickle

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

validating_metrics = {}
samples = list(TRUTH_SET.keys())
causal_sj = list(TRUTH_SET.values())


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


# How to define if a splice junction is present in the filtered data based on the
# 1kb-10kb window?
def is_present_causal_sj(filtered_bed_dat, causal_sj):
    causal_sj_row = filtered_bed_dat[
        (filtered_bed_dat["chr"] == causal_sj.split(":")[0]) &
        ((filtered_bed_dat["start"] == int(causal_sj.split(":")[1].split("-")[0])) |
         (filtered_bed_dat["end"] == int(causal_sj.split(":")[1].split("-")[
                                             1])))]
    print(causal_sj_row)
    return causal_sj_row.shape[0] > 0


def is_present_causal_sj_exact(filtered_bed_dat, causal_sj):
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
    merged_dat = merged_dat[merged_dat["uniquely_mapped"] >= 2]
    merged_dat = merged_dat[merged_dat["maximum_spliced_alignment_overhang"] >= 20]
    return merged_dat


def get_sj_metric(dat, chr, start, end, metric_col):
    return dat[(dat["chr"] == chr) &
               (dat["start"] == start) &
               (dat["end"] == end)][metric_col].iloc[0]


def get_sj_different_between_dats(sample_dat, gtex_dat,
                                  lr_sj_threshold,
                                  lr_coverage_threshold,
                                  cur_sample):
    pd.set_option('display.max_columns', None)
    merged_dat = pd.merge(sample_dat, gtex_dat, how='inner', on=["chr", "start", "end"],
                          suffixes=('', '_y'))
    pseudocount = 1e-12
    sj_ratio = (merged_dat["normalized_by_sj"] + pseudocount) / \
               (merged_dat["normalized_by_sj_y"] + pseudocount)

    coverage_ratio = (merged_dat["normalized_by_coverage"] + pseudocount) / \
                     (merged_dat["normalized_by_coverage_y"] + pseudocount)

    merged_dat["lr_sj"] = abs(np.log2(sj_ratio))
    merged_dat["lr_coverage"] = abs(np.log2(coverage_ratio))

    # # Test
    # pd.set_option('display.max_columns', None)
    # cur_sample = "UC84-1RNA"
    # causal_sj = TRUTH_SET[cur_sample]
    # is_present_causal_sj(merged_dat, causal_sj)
    # #
    # is_present_causal_sj(merged_dat, TRUTH_SET[cur_sample])
    # print(f"{TRUTH_SET[cur_sample]} is not present in {cur_sample}")
    # get the values of dict TRUTH_SET
    for cur_causal_sj in causal_sj:
        if not is_present_causal_sj_exact(merged_dat, cur_causal_sj):
            # print(f"{cur_causal_sj} is not present in {cur_sample}")
            continue

        cur_chr = cur_causal_sj.split(":")[0]
        cur_start = int(cur_causal_sj.split(":")[1].split("-")[0])
        cur_end = int(cur_causal_sj.split(":")[1].split("-")[1])

        cur_splicing_percentage = get_sj_metric(merged_dat,
                                                cur_chr,
                                                cur_start,
                                                cur_end,
                                                "normalized_by_sj")
        cur_coverage = get_sj_metric(merged_dat,
                                     cur_chr,
                                     cur_start,
                                     cur_end,
                                     "normalized_by_coverage")

        cur_lr_sj = get_sj_metric(merged_dat,
                                  cur_chr,
                                  cur_start,
                                  cur_end,
                                  "lr_sj")

        cur_lr_coverage = get_sj_metric(merged_dat,
                                        cur_chr,
                                        cur_start,
                                        cur_end,
                                        "lr_coverage")

        gtex_sj = get_sj_metric(merged_dat,
                                cur_chr,
                                cur_start,
                                cur_end,
                                "normalized_by_sj_y")

        gtex_coverage = get_sj_metric(merged_dat,
                                      cur_chr,
                                      cur_start,
                                      cur_end,
                                      "normalized_by_coverage_y")

        cur_df = pd.DataFrame([[cur_sample, cur_causal_sj, cur_splicing_percentage,
                                cur_coverage, cur_lr_sj, cur_lr_coverage, gtex_sj,
                                gtex_coverage]])
        cur_df.columns = ["sample_id", "sj", "splicing_percentage", "coverage", "lr_sj",
                          "lr_coverage", "gtex_sj", "gtex_coverage"]
        print(cur_sample)
        print(cur_df)

        global validating_metrics
        if cur_causal_sj not in validating_metrics:
            validating_metrics[cur_causal_sj] = cur_df
        else:
            new_df = pd.concat([validating_metrics[cur_causal_sj], cur_df], axis=0)
            validating_metrics[cur_causal_sj] = new_df

    pd.set_option('display.max_columns', None)
    # is_present_causal_sj(merged_dat, TRUTH_SET[cur_sample])
    print("-----------------------------------")
    # merged_dat = merged_dat[(merged_dat["lr_sj"] > lr_sj_threshold) | (
    #         merged_dat["lr_coverage"] > lr_coverage_threshold)]
    merged_dat = merged_dat[(merged_dat["lr_coverage"] > lr_coverage_threshold)]

    merged_dat = merged_dat.drop(
        columns=[col for col in merged_dat.columns if col.endswith("_y")])
    merged_dat = merged_dat.drop(columns=["lr_sj", "lr_coverage"])

    return merged_dat


def normalize_by_adjacent_sj(chr, start, end, compare_dat, rd_normalized_coverage,
                             gene):
    gene_start = int(DISEASE_GENES[gene].split(":")[1].split("-")[0])
    gene_end = int(DISEASE_GENES[gene].split(":")[1].split("-")[1])
    offset = 10000
    compare_dat = compare_dat[(compare_dat["chr"] == chr) &
                              (compare_dat["start"] >= gene_start) &
                              (compare_dat["end"] <= gene_end) &
                              (((compare_dat["start"] >= start - offset) & (
                                          compare_dat["start"] <= start + offset))
                               | ((compare_dat["end"] >= end - offset) & (
                                                  compare_dat["end"] <= end + offset)))]

    if compare_dat.shape[0] == 0:
        return 0
    return abs(np.log2(
        rd_normalized_coverage / np.mean(compare_dat["normalized_by_coverage"])))


def compare_with_gtex(gtex_normalized_dat, dat, lr_sj_threshold,
                      lr_coverage_threshold, cur_sample):
    # in_rd_not_in_gtex_dat = get_sj_only_in_first_dat(dat, gtex_dat)
    in_rd_not_in_gtex_normalized_dat = get_sj_only_in_first_dat(dat,
                                                                gtex_normalized_dat)
    print("Number of splice junctions only in the sample data: ",
          in_rd_not_in_gtex_normalized_dat.shape[0])
    # gtex_mean_normalized_coverage_by_gene = gtex_normalized_dat.groupby("gene")[
    #     "normalized_by_coverage"].mean()
    # gtex_gene_mean_normalized_coverage_dict = gtex_mean_normalized_coverage_by_gene.to_dict()
    # rd_mean_normalized_coverage_by_gene = dat.groupby("gene")[
    #     "normalized_by_coverage"].mean()
    # rd_gene_mean_normalized_coverage_dict = rd_mean_normalized_coverage_by_gene.to_dict()
    #
    # in_rd_not_in_gtex_normalized_dat["lr_coverage"] = abs(np.log2(
    #     in_rd_not_in_gtex_normalized_dat["normalized_by_coverage"] / \
    #     in_rd_not_in_gtex_normalized_dat["gene"].map(
    #         rd_gene_mean_normalized_coverage_dict)))

    in_rd_not_in_gtex_normalized_dat["lr_coverage"] = \
        in_rd_not_in_gtex_normalized_dat.apply(lambda row:
                                               normalize_by_adjacent_sj(row["chr"],
                                                                        row["start"],
                                                                        row["end"],
                                                                        dat,
                                                                        row[
                                                                            "normalized_by_coverage"],
                                                                        row["gene"]),
                                               axis=1)
    novel_threshold = 3

    # Plot the distribution of lr_coverage
    causal_sj = TRUTH_SET[cur_sample]
    cur_chr = causal_sj.split(":")[0]
    cur_start = int(causal_sj.split(":")[1].split("-")[0])
    cur_end = int(causal_sj.split(":")[1].split("-")[1])
    condition = ((in_rd_not_in_gtex_normalized_dat["chr"] == cur_chr) &
                 ((in_rd_not_in_gtex_normalized_dat["start"] == cur_start) |
                  (in_rd_not_in_gtex_normalized_dat["end"] == cur_end)))
    colors = ["#ffd700" if cond else "#adaae1" for cond in condition]
    plt.figure(figsize=(10, 8))
    plt.scatter([i+1 for i in range(in_rd_not_in_gtex_normalized_dat.shape[0])],
                in_rd_not_in_gtex_normalized_dat["lr_coverage"], c=colors)
    plt.xlabel("Splice junctions")
    plt.ylabel("Metric")
    plt.title(f"Novel splice junctions in sample {cur_sample}")
    plt.axhline(y=novel_threshold, color="#8da5c8", linestyle='--', linewidth=2)
    plt.text(x=in_rd_not_in_gtex_normalized_dat.shape[0], y=3, s='y=3', color="#8da5c8",
             ha='right', va='bottom', fontsize=15)
    plt.savefig(f"{cur_sample}_novel_sj.png")


    print(sorted(in_rd_not_in_gtex_normalized_dat["lr_coverage"]))
    pd.set_option('display.max_columns', None)
    is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample])
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
        in_rd_not_in_gtex_normalized_dat["lr_coverage"] <= novel_threshold]
    print("Number of splice junctions only in the sample data after filtering: ",
          in_rd_not_in_gtex_normalized_dat.shape[0])
    print(in_rd_not_in_gtex_normalized_dat)
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat.drop(
        columns=["lr_coverage"])

    # in_gtex_not_in_rd_dat = get_sj_only_in_first_dat(gtex_dat, dat)
    in_gtex_normalized_not_in_rd_dat = get_sj_only_in_first_dat(gtex_normalized_dat,
                                                                dat)
    different_between_rd_and_gtex = get_sj_different_between_dats(dat,
                                                                  gtex_normalized_dat,
                                                                  lr_sj_threshold,
                                                                  lr_coverage_threshold,
                                                                  cur_sample)

    res = pd.concat([in_rd_not_in_gtex_normalized_dat,
                     # in_gtex_normalized_not_in_rd_dat,
                     different_between_rd_and_gtex])
    return res
    # pass
    # diff_dat = pd.concat([in_rd_not_in_gtex_normalized_dat,
    #                       in_gtex_normalized_not_in_rd_dat],
    #                       axis=0, ignore_index=True)
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


def count_reads_with_same_donor_or_acceptor(reads_with_same_donor,
                                            reads_with_same_acceptor,
                                            chr, start, end, uniquely_mapped):
    reads_with_same_donor = \
        reads_with_same_donor[(reads_with_same_donor["chr"] == chr) &
                              (reads_with_same_donor["start"] == start)][
            "total_reads"].iloc[0]
    reads_with_same_acceptor = \
        reads_with_same_acceptor[(reads_with_same_acceptor["chr"] == chr)
                                 & (reads_with_same_acceptor["end"] == end)][
            "total_reads"].iloc[0]

    # The number of uniquely_mapped reads are counted twice in both
    # reads_with_same_donor and reads_with_same_acceptor. Therefore, subtract it once
    # in the denominator.
    return uniquely_mapped / (
            reads_with_same_donor + reads_with_same_acceptor - uniquely_mapped)


def normalize_reads_by_gene(dat, coverage_bw):
    """
    Normalize the number of reads by dividing the reads of a splice junction by the
    total number of reads of all splice junctions within the gene. Also, normalize the
    reads by the coverage of the gene.
    """

    # Normalize by total reads of splice junctions within the gene.
    # gene_total_sj_reads = dat.groupby("gene")["uniquely_mapped"].sum().reset_index()
    # gene_total_sj_reads.columns = ["gene", "total_reads"]
    # gene_total_sj_reads_dict = {}
    # for i in range(gene_total_sj_reads.shape[0]):
    #     gene_total_sj_reads_dict[gene_total_sj_reads.iloc[i, 0]] = \
    #         gene_total_sj_reads.iloc[i, 1]
    reads_with_same_donor = dat.groupby(["chr", "start"])[
        "uniquely_mapped"].sum().reset_index()
    reads_with_same_donor.columns = ["chr", "start", "total_reads"]

    reads_with_same_acceptor = dat.groupby(["chr", "end"])[
        "uniquely_mapped"].sum().reset_index()
    reads_with_same_acceptor.columns = ["chr", "end", "total_reads"]
    dat["normalized_by_sj"] = dat.apply(lambda row:
                                        count_reads_with_same_donor_or_acceptor(
                                            reads_with_same_donor,
                                            reads_with_same_acceptor,
                                            row["chr"],
                                            row["start"],
                                            row["end"],
                                            row["uniquely_mapped"]),
                                        axis=1)

    # print(gene_total_sj_reads_dict)
    # dat["normalized_by_sj"] = dat["uniquely_mapped"] / dat["gene"].map(
    #     gene_total_sj_reads_dict)

    # Normalize by the coverage of the gene.
    mean_gene_coverage_dict = {}
    for gene in DISEASE_GENES:
        gene_chr = DISEASE_GENES[gene].split(":")[0]
        gene_start = int(DISEASE_GENES[gene].split(":")[1].split("-")[0])
        gene_end = int(DISEASE_GENES[gene].split(":")[1].split("-")[1])

        total_exon_coverage, num_of_bases = get_exon_coverage(coverage_bw,
                                                              gene_chr,
                                                              gene_start,
                                                              gene_end,
                                                              gene)
        mean_gene_coverage_dict[gene] = total_exon_coverage / num_of_bases
    #     total_intron_coverage, num_of_bases = get_intron_coverage(coverage_bw,
    #                                                               gene_chr,
    #                                                               gene_start,
    #                                                               gene_end)
    #     mean_intron_coverage_dict[gene] = total_intron_coverage / num_of_bases
    #
    # print(mean_gene_coverage_dict)
    # dat["sj_coverage"] = dat.apply(lambda row: np.nanmean([i for i in list(
    #     coverage_bw.values(
    #         row["chr"],
    #         row["start"],
    #         row["end"])) if i != 0]),
    #     axis=1)

    dat["normalized_by_coverage"] = dat["uniquely_mapped"] / dat["gene"].map(
        mean_gene_coverage_dict)
    # pd.set_option('display.max_columns', None)
    # print("-----------------------------------")
    # dat["normalized_by_coverage"] = dat["sj_coverage"] / dat["gene"].map(
    #     mean_intron_coverage_dict)
    # dat["normalized_by_coverage"] = dat["uniquely_mapped"] / dat.apply(lambda row:
    #                                                                    get_mean_adjacent_exon_coverage(
    #                                                                        coverage_bw,
    #                                                                        row["chr"],
    #                                                                        row["start"],
    #                                                                        row["end"]),
    #                                                                    axis=1)
    # print(mean_gene_coverage_dict["RYR1"])
    # print(mean_intron_coverage_dict["RYR1"])

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

    cur_mane_exons = cur_mane_exons.drop_duplicates(subset=["chr", "start", "end"])

    total_exon_coverage = 0
    num_of_bases = 0
    for i in range(cur_mane_exons.shape[0]):
        chr = cur_mane_exons.iloc[i, 0]
        start = cur_mane_exons.iloc[i, 1]
        end = cur_mane_exons.iloc[i, 2]
        num_of_bases += end - start

        coverage = list(bw.values(chr, start, end))
        total_exon_coverage += np.nansum(coverage)

    return total_exon_coverage, num_of_bases


def get_mean_adjacent_exon_coverage(bw, chr, sj_start, sj_end):
    cur_chr_disease_genes = mane_exons_in_disease_genes[
        mane_exons_in_disease_genes["chr"] == chr]
    exon1 = cur_chr_disease_genes[cur_chr_disease_genes["end"] == sj_start]

    if exon1.shape[0] > 1:
        raise Exception(f"More than one exon found ending at {chr}:{sj_start}.")
    exon1_start = list(exon1["start"])[0]
    exon1_end = list(exon1["end"])[0]
    exon1_coverage = np.nansum(list(bw.values(chr, exon1_start, exon1_end)))

    exon2 = cur_chr_disease_genes[cur_chr_disease_genes["start"] == sj_end]
    if exon2.shape[0] > 1:
        raise Exception(f"More than one exon found starting at {chr}:{sj_end}.")
    exon2_start = list(exon2["start"])[0]
    exon2_end = list(exon2["end"])[0]
    exon2_coverage = np.nansum(list(bw.values(chr, exon2_start, exon2_end)))

    return (exon1_coverage + exon2_coverage) / 2


def get_intron_coverage(bw, chr, start, end, gene_name=None):
    exon_coverage, num_of_bases = get_exon_coverage(bw, chr, start, end, gene_name)
    total_coverage = np.nansum(bw.values(chr, start, end))
    return total_coverage - exon_coverage, end - start - num_of_bases


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
                                         lr_coverage_threshold=5,
                                         cur_sample=sample_id)
    print(filtered_bed_dat.shape)
    grouped_gene = filtered_bed_dat.groupby("gene").size()

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
    captured_samples = pd.DataFrame(columns=["sample_id", "num_sj_candidates",
                                             "captured"])
    number_of_candidates = []
    # gtex_dat = read_junctions_bed(GTEx, USE_COLS)
    # gtex_dat = filter_by_uniquely_mapped_reads(gtex_dat, 1)

    gtex_normalized_dat = read_junctions_bed(GTEx_NORMALIZED_BED, USE_COLS)
    gtex_normalized_dat = filter_by_uniquely_mapped_reads(gtex_normalized_dat, 1)
    for cur_sample in TRUTH_SET:
        # if cur_sample != "205E_BD_M1":
        #     continue
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
            row_data = [cur_sample, filtered_bed_dat.shape[0], True]
        else:
            row_data = [cur_sample, filtered_bed_dat.shape[0], False]
        new_row = pd.DataFrame([row_data], columns=captured_samples.columns)
        captured_samples = pd.concat([captured_samples, new_row], ignore_index=True)

    print(captured_samples[captured_samples['captured'] == True])
    print(f"In total, "
          f"{captured_samples[captured_samples['captured'] == True].shape[0]} out of"
          f" {len(TRUTH_SET)} causal splice junctions are captured in the truth set.")
    print(captured_samples)
    print(np.mean(number_of_candidates), np.median(number_of_candidates))

    captured_samples = captured_samples.sort_values(by="num_sj_candidates",
                                                    ascending=False)
    plot_conditions = (captured_samples["captured"] == True)
    plot_colors = ["#c8dbb9" if cond else "#c85454" for cond in plot_conditions]
    plt.figure(figsize=(12, 10))
    plt.bar(captured_samples["sample_id"], captured_samples["num_sj_candidates"],
            color=plot_colors)
    plt.xlabel("Sample")
    plt.ylabel("Number of splice junction candidates")
    plt.ylim(0, 100)
    plt.title("Number of splice junction candidates for each sample in truth set")
    plt.axhline(y=np.mean(number_of_candidates), color="#8da5c8", linestyle='--')
    plt.text(x=captured_samples.shape[0], y=np.mean(number_of_candidates), s=f'y={np.mean(number_of_candidates)}', color="#8da5c8",
             ha='right', va='bottom', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.savefig("num_sj_candidates.png")


if __name__ == "__main__":
    # bigwig_path = "gs://tgg-rnaseq/batch_0/bigWig/210DB_BW_M1.bigWig"
    # bw = read_coverage_bigwig(bigwig_path)
    # values = bw.values("chr19", 1, 3)
    # print(values)
    mane_exons = pd.read_csv("MANE_exon_1based_start.tsv", sep="\t")
    mane_exons_in_disease_genes = filter_to_disease_genes(mane_exons, DISEASE_GENES)
    # print(mane_exons_in_disease_genes)
    # mane_exons_in_disease_genes = mane_exons_in_disease_genes[
    #     mane_exons_in_disease_genes["chr"] == "chr19"]
    # print(mane_exons_in_disease_genes)
    # test = mane_exons_in_disease_genes[mane_exons_in_disease_genes["end"] == 38467812]
    # print(test)
    # test2 = mane_exons_in_disease_genes[
    #     mane_exons_in_disease_genes["start"] == 38468965]
    # print(test2)

    # Key: causal SJ, Value: a DataFrame with cols [sample_id, splicing_ratio, reads_by_exon_coverage]
    # print(samples)
    # print(causal_sj)
    main()
    # print(validating_metrics)
    # with open('validating_metrics_full.pkl', 'wb') as pickle_file:
    #     pickle.dump(validating_metrics, pickle_file)

    # with open('validating_metrics_full.pkl', 'rb') as file:
    #     validating_metrics = pickle.load(file)
    #
    # print(validating_metrics)
    #
    # for key in validating_metrics:
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(validating_metrics[key]["sample_id"],
    #                 validating_metrics[key]["splicing_percentage"])
    #     plt.title(f"{key}_splicing_percentage")
    #     plt.xlabel("sample")
    #     plt.ylabel("splicing_percentage")
    #     for i, label in enumerate(validating_metrics[key]["sample_id"]):
    #         plt.annotate(label, (list(validating_metrics[key]["sample_id"])[i],
    #                              list(validating_metrics[key]["splicing_percentage"])[i]),
    #                      textcoords="offset points",
    #                      xytext=(0, 10), ha='center')
    #
    #     plt.savefig(f"{key}_splicing_percentage.png")
    #
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(validating_metrics[key]["sample_id"],
    #                 validating_metrics[key]["coverage"])
    #     plt.title(f"{key}_coverage")
    #     plt.xlabel("sample")
    #     plt.ylabel("coverage")
    #     for i, label in enumerate(validating_metrics[key]["sample_id"]):
    #         plt.annotate(label, (list(validating_metrics[key]["sample_id"])[i],
    #                              list(validating_metrics[key]["coverage"])[i]),
    #                      textcoords="offset points",
    #                      xytext=(0, 10), ha='center')
    #     plt.savefig(f"{key}_coverage.png")
    #
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(validating_metrics[key]["sample_id"],
    #                 validating_metrics[key]["lr_sj"])
    #     plt.title(f"{key}_lr_sj")
    #     plt.xlabel("sample")
    #     plt.ylabel("lr_sj")
    #     for i, label in enumerate(validating_metrics[key]["sample_id"]):
    #         plt.annotate(label, (list(validating_metrics[key]["sample_id"])[i],
    #                              list(validating_metrics[key]["lr_sj"])[i]),
    #                      textcoords="offset points",
    #                      xytext=(0, 10), ha='center')
    #     plt.savefig(f"{key}_lr_sj.png")
    #
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(validating_metrics[key]["sample_id"],
    #                 validating_metrics[key]["lr_coverage"])
    #     plt.title(f"{key}_lr_coverage")
    #     plt.xlabel("sample")
    #     plt.ylabel("lr_coverage")
    #     for i, label in enumerate(validating_metrics[key]["sample_id"]):
    #         plt.annotate(label, (list(validating_metrics[key]["sample_id"])[i],
    #                              list(validating_metrics[key]["lr_coverage"])[i]),
    #                      textcoords="offset points",
    #                      xytext=(0, 10), ha='center')
    #     plt.savefig(f"{key}_lr_coverage.png")

    # bigwig_path = f"gs://tgg-rnaseq/batch_0/bigWig/CLA_180CJ_DP_2.bigWig"
    # bw = read_coverage_bigwig(bigwig_path)
    # c, b = get_intron_coverage(bw, "chr19", 38468964, 38469140)
    # # c, b = get_intron_coverage(bw, "chr19", 38468376, 38468377)
    # print(c, b, c/b)
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
