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

UNIQUELY_MAPPED = 2
MAXIMUM_OVERHANG = 20
LR_COVERAGE_THRESHOLD = 5
PERCENTAGE_THRESHOLD = 0.7
NOVEL_SJ_THRESHOLD = 3

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


def get_sj_only_in_first_dat(dat1, dat2, gtex):
    merged_dat = pd.merge(dat1, dat2, how="left", on=["chr", "start", "end"],
                          suffixes=('', '_y'),
                          indicator=True).query('_merge == "left_only"').drop(
        columns=['_merge'])
    merged_dat = merged_dat.drop(
        columns=[col for col in merged_dat.columns if col.endswith("_y")])
    if gtex:
        merged_dat = merged_dat[merged_dat["uniquely_mapped"] >= UNIQUELY_MAPPED]
        merged_dat = merged_dat[merged_dat["num_samples_with_this_junction"] /
                                merged_dat["num_samples_total"] >= PERCENTAGE_THRESHOLD]
        merged_dat = merged_dat[merged_dat["maximum_spliced_alignment_overhang"] >= MAXIMUM_OVERHANG]
    else:
        merged_dat = merged_dat[merged_dat["uniquely_mapped"] >= UNIQUELY_MAPPED]
        merged_dat = merged_dat[merged_dat["maximum_spliced_alignment_overhang"] >=
                                MAXIMUM_OVERHANG]
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
    merged_dat = merged_dat[(merged_dat["lr_coverage"] >= lr_coverage_threshold)]

    merged_dat = merged_dat.drop(
        columns=[col for col in merged_dat.columns if col.endswith("_y")])
    # merged_dat = merged_dat.drop(columns=["lr_sj", "lr_coverage"])

    return merged_dat


def normalize_by_adjacent_sj(chr, start, end, compare_dat, rd_normalized_coverage,
                             gene):
    # TODO: include splice junctions that are in unannotated gene regions
    gene_start = int(DISEASE_GENES[gene].split(":")[1].split("-")[0])
    gene_end = int(DISEASE_GENES[gene].split(":")[1].split("-")[1])
    offset = 10000
    compare_dat = compare_dat[(compare_dat["chr"] == chr) &
                              (compare_dat["start"] >= gene_start) &
                              (compare_dat["end"] <= gene_end) &
                              ((compare_dat["start"] >= start - offset) &
                              ((compare_dat["end"] <= end + offset)))]

    if compare_dat.shape[0] == 0:
        return 0
    print("check denominator")
    print(np.mean(compare_dat["normalized_by_coverage"]))
    print(compare_dat)
    return abs(np.log2(
        rd_normalized_coverage / np.mean(compare_dat["normalized_by_coverage"])))


def compare_with_gtex(gtex_normalized_dat, dat, lr_sj_threshold,
                      lr_coverage_threshold, cur_sample):
    # in_rd_not_in_gtex_dat = get_sj_only_in_first_dat(dat, gtex_dat)
    in_rd_not_in_gtex_normalized_dat = get_sj_only_in_first_dat(dat,
                                                                gtex_normalized_dat,
                                                                False)
    print("test1:")
    print(f"Number of splice junctions only in the sample data: {in_rd_not_in_gtex_normalized_dat.shape[0]}")
    in_gtex_normalized_not_in_rd_dat = get_sj_only_in_first_dat(gtex_normalized_dat,
                                                                dat,
                                                                True)
    print("Number of splice junctions only in the GTEx data: ",
          in_gtex_normalized_not_in_rd_dat.shape[0])
    in_gtex_normalized_not_in_rd_dat["lr_coverage"] = \
        in_gtex_normalized_not_in_rd_dat.apply(lambda row:
                                               normalize_by_adjacent_sj(row["chr"],
                                                                        row["start"],
                                                                        row["end"],
                                                                        gtex_normalized_dat,
                                                                        row[
                                                                            "normalized_by_coverage"],
                                                                        row["gene"]),
                                               axis=1)
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
    pd.set_option('display.max_columns', None)
    # novel_threshold = 3

    # Plot the distribution of lr_coverage
    # causal_sj = TRUTH_SET[cur_sample]
    # cur_chr = causal_sj.split(":")[0]
    # cur_start = int(causal_sj.split(":")[1].split("-")[0])
    # cur_end = int(causal_sj.split(":")[1].split("-")[1])
    # condition = ((in_rd_not_in_gtex_normalized_dat["chr"] == cur_chr) &
    #              ((in_rd_not_in_gtex_normalized_dat["start"] == cur_start) |
    #               (in_rd_not_in_gtex_normalized_dat["end"] == cur_end)))
    # colors = ["#ffd700" if cond else "#adaae1" for cond in condition]
    # plt.figure(figsize=(10, 8))
    # plt.scatter([i + 1 for i in range(in_rd_not_in_gtex_normalized_dat.shape[0])],
    #             in_rd_not_in_gtex_normalized_dat["lr_coverage"], c=colors)
    # plt.xlabel("Splice junctions")
    # plt.ylabel("Metric")
    # plt.title(f"Novel splice junctions in sample {cur_sample}")
    # plt.axhline(y=novel_threshold, color="#8da5c8", linestyle='--', linewidth=2)
    # plt.text(x=in_rd_not_in_gtex_normalized_dat.shape[0], y=3, s='y=3', color="#8da5c8",
    #          ha='right', va='bottom', fontsize=15)
    # plt.savefig(f"{cur_sample}_novel_sj.png")

    print(sorted(in_rd_not_in_gtex_normalized_dat["lr_coverage"]))
    pd.set_option('display.max_columns', None)
    is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample])
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
        in_rd_not_in_gtex_normalized_dat["lr_coverage"] <= NOVEL_SJ_THRESHOLD]


    in_gtex_normalized_not_in_rd_dat = in_gtex_normalized_not_in_rd_dat[
        in_gtex_normalized_not_in_rd_dat["lr_coverage"] <= NOVEL_SJ_THRESHOLD]

    print("Number of splice junctions only in the sample data after filtering: ",
          in_rd_not_in_gtex_normalized_dat.shape[0])
    print(in_rd_not_in_gtex_normalized_dat)

    # in_gtex_not_in_rd_dat = get_sj_only_in_first_dat(gtex_dat, dat)
    # in_gtex_normalized_not_in_rd_dat = get_sj_only_in_first_dat(gtex_normalized_dat,
    #                                                             dat)
    different_between_rd_and_gtex = get_sj_different_between_dats(dat,
                                                                  gtex_normalized_dat,
                                                                  lr_sj_threshold,
                                                                  lr_coverage_threshold,
                                                                  cur_sample)

    if is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]):
        print("here1")
        rank = get_rank_of_causal_sj(in_rd_not_in_gtex_normalized_dat,
                                     TRUTH_SET[cur_sample],
                                     "lr_coverage", True)
    elif is_present_causal_sj(in_gtex_normalized_not_in_rd_dat, TRUTH_SET[cur_sample]):
        print("here2")
        rank = get_rank_of_causal_sj(in_gtex_normalized_not_in_rd_dat,
                                     TRUTH_SET[cur_sample],
                                     "lr_coverage", True)
    elif is_present_causal_sj(different_between_rd_and_gtex, TRUTH_SET[cur_sample]):
        print("here3")
        rank = get_rank_of_causal_sj(different_between_rd_and_gtex,
                                     TRUTH_SET[cur_sample],
                                     "lr_coverage", False)
    else:
        rank = -1

    in_gtex_normalized_not_in_rd_dat = in_gtex_normalized_not_in_rd_dat.drop(
        columns=["lr_coverage"])
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat.drop(
        columns=["lr_coverage"])
    different_between_rd_and_gtex = different_between_rd_and_gtex.drop(columns=[
        "lr_sj", "lr_coverage"])

    res = pd.concat([
                     in_rd_not_in_gtex_normalized_dat,
                     in_gtex_normalized_not_in_rd_dat,
                     different_between_rd_and_gtex
    ])
    print(f"In sample but not in GTEx: {in_rd_not_in_gtex_normalized_dat.shape[0]}")
    print(f"In GTEx but not in sample: {in_gtex_normalized_not_in_rd_dat.shape[0]}")
    print(f"In both: {different_between_rd_and_gtex.shape[0]}")
    return res, rank


def get_rank_of_causal_sj(dat, causal_sj, col, ascending):
    cur_chr = causal_sj.split(":")[0]
    cur_start = int(causal_sj.split(":")[1].split("-")[0])
    cur_end = int(causal_sj.split(":")[1].split("-")[1])
    print(dat)
    print(cur_chr, cur_start, cur_end)
    print(causal_sj)
    dat = dat.sort_values(by=col, ascending=ascending)
    dat = dat.reset_index(drop=True)
    matching_start_index = dat[(dat["chr"] == cur_chr) &
                               (dat["start"] == cur_start)].index
    if len(matching_start_index) > 0:
        matching_start_index = matching_start_index[0]
    else:
        matching_start_index = float("inf")

    matching_end_index = dat[(dat["chr"] == cur_chr) &
                             (dat["end"] == cur_end)].index
    if len(matching_end_index) > 0:
        matching_end_index = matching_end_index[0]
    else:
        matching_end_index = float("inf")
    matching_index = min(matching_start_index, matching_end_index)
    return matching_index + 1


def read_junctions_bed(junctions_bed_path, use_cols=None):
    bed_dat = pd.read_csv(junctions_bed_path, sep="\t", header=None)
    bed_dat.columns = ["chr", "start", "end", "description", "uniquely_mapped",
                       "strand"]
    bed_dat = bed_dat[bed_dat["chr"].isin(CHROM_SET)]
    bed_dat = bed_dat.drop(columns=["uniquely_mapped"])

    description_cols = bed_dat["description"].iloc[0].split(";")
    description_cols = [col.split("=")[0] for col in description_cols]

    bed_dat[description_cols] = bed_dat[
        "description"].str.split(";", expand=True)
    # TODO: calculate theses columns if they are not present in the description.
    if "num_samples_with_this_junction" not in description_cols:
        bed_dat["num_samples_with_this_junction"] = 0
        bed_dat["num_samples_total"] = 0
    bed_dat = bed_dat.drop(columns=["description"])
    for cur_col in description_cols:
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


def get_mean_exon_covarege_within_a_gene_window(bw, sj_chr, sj_start, sj_end,
                                                window_size,
                                                gene_name):
    cur_mane_exons = mane_exons_in_disease_genes[
        mane_exons_in_disease_genes["gene"] == gene_name]
    pd.set_option('display.max_columns', None)
    cur_mane_exons = cur_mane_exons[(cur_mane_exons["chr"] == sj_chr) &
                                    (cur_mane_exons["start"] >= sj_start - window_size)
                                    & (cur_mane_exons["end"] <= sj_end + window_size)]

    cur_mane_exons = cur_mane_exons.drop_duplicates(subset=["chr", "start", "end"])
    if cur_mane_exons.shape[0] == 0:
        return 1e-6

    num_of_bases = 0
    total_exon_coverage = 0
    for i in range(cur_mane_exons.shape[0]):
        cur_chr = cur_mane_exons.iloc[i, 0]
        cur_start = cur_mane_exons.iloc[i, 1]
        cur_end = cur_mane_exons.iloc[i, 2]
        num_of_bases += cur_end - cur_start

        coverage = list(bw.values(cur_chr, cur_start, cur_end))
        total_exon_coverage += np.nansum(coverage)

    return total_exon_coverage / num_of_bases


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

    print("mean coverage")
    print(mean_gene_coverage_dict["PYROXD1"])
    dat["normalized_by_coverage"] = dat["uniquely_mapped"] / dat["gene"].map(
        mean_gene_coverage_dict)
    # window_size = 3000
    # dat["normalized_by_coverage"] = dat["uniquely_mapped"] / dat.apply(lambda row:
    #                                                                    get_mean_exon_covarege_within_a_gene_window(coverage_bw,
    #                                                                                                                row["chr"],
    #                                                                                                                row["start"],
    #                                                                                                                row["end"],
    #                                                                                                                window_size,
    #                                                                                                                row["gene"]),
    #                                                                       axis=1)
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
    # TODO: use more sophisiticated way to get the coverage of the gene rather than
    #  mane exons.
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
        cur_chr = cur_mane_exons.iloc[i, 0]
        cur_start = cur_mane_exons.iloc[i, 1]
        cur_end = cur_mane_exons.iloc[i, 2]
        num_of_bases += cur_end - cur_start

        coverage = list(bw.values(cur_chr, cur_start, cur_end))
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
    # print("number here1")
    print(bed_dat.shape)
    print(bed_dat.columns)

    # 2. Filter to known disease genes.
    filtered_bed_dat = filter_to_disease_genes(bed_dat, DISEASE_GENES)
    filtered_gtex_normalized_dat = filter_to_disease_genes(gtex_normalized_dat,
                                                           DISEASE_GENES)

    # print("number here2")
    print(filtered_bed_dat.shape)
    # causal_sj = TRUTH_SET[sample_id]
    # is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)

    # 3. Compare reads and coverage with GTEx data.
    # Get the coverage file.
    bw = read_coverage_bigwig(bigwig_path)
    gtex_bw = read_coverage_bigwig(GTEx_NORMALIZED_BIGWIG)

    normalized_filtered_bed_dat = normalize_reads_by_gene(filtered_bed_dat, bw)
    normalized_gtex_normalized_dat = normalize_reads_by_gene(
        filtered_gtex_normalized_dat, gtex_bw)
    filtered_bed_dat, rank = compare_with_gtex(normalized_gtex_normalized_dat,
                                               normalized_filtered_bed_dat,
                                               lr_sj_threshold=2,
                                               lr_coverage_threshold=LR_COVERAGE_THRESHOLD,
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
    return filtered_bed_dat, rank


def main():
    captured_samples = pd.DataFrame(columns=["sample_id", "num_sj_candidates",
                                             "captured", "rank"])
    number_of_candidates = []
    # gtex_dat = read_junctions_bed(GTEx, USE_COLS)
    # gtex_dat = filter_by_uniquely_mapped_reads(gtex_dat, 1)

    gtex_normalized_dat = read_junctions_bed(GTEx_NORMALIZED_BED, USE_COLS)
    gtex_normalized_dat = filter_by_uniquely_mapped_reads(gtex_normalized_dat, 1)
    for cur_sample in TRUTH_SET:
        if cur_sample != "UC84-1RNA":
            continue
        if cur_sample in BATCH_2023:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_2023_01/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
            bigwig_path = f"gs://tgg-rnaseq/batch_2023_01/bigWig/{cur_sample}.bigWig"
        elif cur_sample in BATCH_2022:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_2022_01/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
            bigwig_path = f"gs://tgg-rnaseq/batch_2022_01/bigWig/{cur_sample}.bigWig"
        else:
            junctions_bed_path = f"gs://tgg-rnaseq/batch_0/junctions_bed_for_igv_js/{cur_sample}.junctions.bed.gz"
            bigwig_path = f"gs://tgg-rnaseq/batch_0/bigWig/{cur_sample}.bigWig"

        bed_dat = read_junctions_bed(junctions_bed_path, USE_COLS)
        filtered_bed_dat, rank = apply_filters(bed_dat, cur_sample,
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
            row_data = [cur_sample, filtered_bed_dat.shape[0], True, rank]
        else:
            row_data = [cur_sample, filtered_bed_dat.shape[0], False, rank]
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
    # captured_samples.to_csv("captured_samples.csv", index=False)

    # captured_samples = pd.read_csv("captured_samples.csv")
    print(captured_samples)
    number_of_candidates = list(captured_samples["num_sj_candidates"])
    plot_conditions = (captured_samples["captured"] == True)
    plot_colors = ["#c8dbb9" if cond else "#c85454" for cond in plot_conditions]
    plt.figure(figsize=(20, 25))
    plt.barh(captured_samples["sample_id"], captured_samples["num_sj_candidates"],
             color=plot_colors)
    plt.xlabel("Number of splice junction candidates", fontsize=50)
    plt.ylabel("Samples", fontsize=50, rotation=0, ha='right', va='center')
    # plt.yticks(fontsize=30)
    plt.yticks([])
    plt.xticks(fontsize=30)
    plt.xlim(0, 150)
    plt.axvline(x=np.mean(number_of_candidates), color="#8da5c8", linestyle='--',
                linewidth=5)
    # plt.text(x=np.mean(number_of_candidates) * 3.8,
    #          y=captured_samples.shape[0] - 3,
    #          s=f'average number of candidates\n='
    #            f'{round(np.mean(number_of_candidates), 2)}',
    #          color="#000000",
    #          ha='right', va='bottom', fontsize=45, weight='bold')

    plt.savefig("num_sj_candidates_2.png", bbox_inches='tight')

    captured_samples = captured_samples[captured_samples["rank"] > 0]
    captured_samples = captured_samples.sort_values(by="rank", ascending=False)
    captured_samples.to_csv("captured_samples_heuristic.csv", index=False)

    fraser2_benchmark_dat = pd.read_csv("fraser2_best_results.txt", sep="\t")
    fraser2_benchmark_dat.columns = ["sample", "called_by_fraser2",
                                     "num_of_sj_candidates", "rank"]
    fraser2_benchmark_dat_called = set(list(fraser2_benchmark_dat[fraser2_benchmark_dat[
        "rank"]>0]["sample"]))
    print(fraser2_benchmark_dat_called)
    print(len(fraser2_benchmark_dat_called))
    captured_samples["called_by_fraser2"] = captured_samples["sample_id"].apply(
        lambda x: "Yes" if x in fraser2_benchmark_dat_called else "No")
    print(captured_samples)
    plot_conditions = (captured_samples["called_by_fraser2"] == "Yes")
    plot_colors = ["#8da5c8" if cond else "#c8dbb9" for cond in plot_conditions]


    plt.figure(figsize=(20, 25))
    plt.barh(captured_samples["sample_id"], captured_samples["rank"],
             color=plot_colors)
    plt.xlabel("Rank of Causal Splice Junctions", fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(np.arange(0, 19, 1), fontsize=30)
    plt.axvline(x=np.mean(captured_samples["rank"]), color="#8da5c8", linestyle='--',
                linewidth=5)
    plt.text(x=np.mean(captured_samples["rank"]) * 4,
             y=captured_samples.shape[0] - 3,
             s=f'average rank of\ncausal splice junctions\n'
               f'={round(np.mean(captured_samples["rank"]), 2)}',
             color="#000000",
             ha='right', va='bottom', fontsize=45, weight='bold')
    plt.savefig("rank.png", bbox_inches='tight')


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
