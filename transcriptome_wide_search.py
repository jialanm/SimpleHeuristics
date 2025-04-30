import os

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pickle
import time
import json
import hail as hl
import requests

from set_heuristics import TRUTH_SET, GTEx, GTEx_NORMALIZED_BED, GTEx_NORMALIZED_BIGWIG, \
    CHROM_SET, DESCRIPTION_COLS, USE_COLS, \
    read_junctions_bed, filter_by_uniquely_mapped_reads, read_coverage_bigwig, \
    is_present_causal_sj, get_sj_only_in_first_dat, BATCH_2023, BATCH_2022
from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import (
    read_from_airtable, RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)

LOCAL_GENE_SIZE = 100000
NOVEL_SJ_THRESHOLD = 1.5
MUSCLE_SAMPLE_TOTAL = 194
SAMPLE_USE_COLS = ["chr", "start", "end", "strand", "uniquely_mapped",
                   "maximum_spliced_alignment_overhang"]

UNIQUE_READS_THRESHOLD = 7

def get_genome_interval(chrom, start, end):
    return f"{chrom}:{start + 1}-{end}"


def get_interval_percentile(dat, chrom, start, end, percentile, window_size):
    dat = dat[(dat["chr"] == chrom) &
              (dat["start"] >= start - window_size) &
              (dat["end"] <= end + window_size)]
    return get_percentile(list(dat["uniquely_mapped"]), percentile)


def get_exon_coverage(bw, chrom, start, end):
    cur_exons = gencode_v47_longest_exons[(gencode_v47_longest_exons["chr"] == chrom) &
                                          (gencode_v47_longest_exons[
                                               "start"] >= start) &
                                          (gencode_v47_longest_exons["end"] <= end)]

    total_exon_coverage = 0
    num_of_bases = 0
    for i in range(cur_exons.shape[0]):
        cur_chr = cur_exons.iloc[i, 0]
        cur_start = cur_exons.iloc[i, 1]
        cur_end = cur_exons.iloc[i, 2]
        num_of_bases += cur_end - cur_start

        coverage = list(bw.values(cur_chr, cur_start, cur_end))
        total_exon_coverage += np.nansum(coverage)

    print(bw.intervals(cur_chr, cur_start, cur_end))
    return total_exon_coverage, num_of_bases


def get_gene_coords_dict(gencode_genes):
    gene_coords = {}
    for i in range(gencode_genes.shape[0]):
        cur_chr = gencode_genes.iloc[i, 0]
        cur_start = gencode_genes.iloc[i, 1]
        cur_end = gencode_genes.iloc[i, 2]
        cur_gene = gencode_genes.iloc[i, 3]
        gene_coords[cur_gene] = f"{cur_chr}:{cur_start}-{cur_end}"
    return gene_coords


def find_closest(nums, A):
    # Handle the edge case where the list is empty
    if not nums:
        raise ValueError("The list is empty.")

    low, high = 0, len(nums) - 1
    closest_index = 0

    while low <= high:
        mid = (low + high) // 2

        # Compare the current element with the target value A
        if abs(nums[mid] - A) < abs(nums[closest_index] - A):
            closest_index = mid

        # Binary search logic
        if nums[mid] < A:
            low = mid + 1
        elif nums[mid] > A:
            high = mid - 1
        else:
            # If we find the exact match, return its index
            return mid

    return closest_index


def find_closest_gene_to_a_sj(gene, chrom, start, end, coords_gene_start_dict,
                              coords_gene_end_dict):
    if gene in all_genes:
        return gene
    # TODO: add a module to handle gene fusion detection.
    start_coords_list = coords_gene_start_dict[chrom][0]
    start_gene_list = coords_gene_start_dict[chrom][1]
    closest_gene_based_on_start_pos = start_gene_list[find_closest(start_coords_list,
                                                                   start)]

    end_coords_list = coords_gene_end_dict[chrom][0]
    end_gene_list = coords_gene_end_dict[chrom][1]
    closest_gene_based_on_end_pos = end_gene_list[find_closest(end_coords_list,
                                                               start)]

    # if closest_gene_based_on_start_pos != closest_gene_based_on_end_pos:
    #     print("The closest gene based on start position is different from the closest gene based on end position")
    #     print(f"Check for gene fusion events at {chr}:{start}-{end}")
    return closest_gene_based_on_start_pos


# TODO: remove `=None` from the following function.
def normalize_reads_by_gene(dat, cur_gene_coverage_dict=None):
    """
    Normalize the number of reads by dividing the reads of a splice junction by the
    coverage of the gene.
    """
    # If a splice junction is in an unannotated region, find its closest gene.

    dat["gene_with_imputation"] = dat.apply(lambda row:
                                            find_closest_gene_to_a_sj(row["gene"],
                                                                      row["chr"],
                                                                      row["start"],
                                                                      row["end"],
                                                                      coords_gene_start_dict,
                                                                      coords_gene_end_dict),
                                            axis=1)
    # TODO: uncomment the following line.
    # dat["normalized_by_coverage"] = dat.apply(lambda row: (row["uniquely_mapped"] /
    #                                                       cur_gene_coverage_dict[row["gene_with_imputation"]])
    #                                                       if cur_gene_coverage_dict[row["gene_with_imputation"]] != 0 else 1e6,
    #                                           axis=1)
    dat["gene"] = dat["gene_with_imputation"]
    dat.drop(columns=["gene_with_imputation"], inplace=True)
    return dat


def get_sj_different_between_dats(sample_dat, gtex_dat,
                                  lr_coverage_threshold):
    pd.set_option('display.max_columns', None)
    merged_dat = pd.merge(sample_dat, gtex_dat, how='inner', on=["chr", "start", "end"],
                          suffixes=('', '_y'))
    pseudocount = 1e-12

    coverage_ratio = (merged_dat["normalized_by_coverage"] + pseudocount) / \
                     (merged_dat["normalized_by_coverage_y"] + pseudocount)

    merged_dat["lr_coverage"] = abs(np.log2(coverage_ratio))

    merged_dat = merged_dat[(merged_dat["lr_coverage"] >= lr_coverage_threshold)]

    merged_dat = merged_dat.drop(
        columns=[col for col in merged_dat.columns if col.endswith("_y")])

    return merged_dat


def compare_with_adjacent_sj(dat, chrom, start, end, uniquely_mapped,
                             gene,window_size=5000):
    # Filter out SJs with less than 10% of the uniquely mapped reads.
    gene_coords = gene_coords_dict[gene]
    gene_start = int(gene_coords.split(":")[1].split("-")[0])
    gene_end = int(gene_coords.split(":")[1].split("-")[1])
    new_start = max(start - window_size, gene_start)
    new_end = min(end + window_size, gene_end)

    # Include a SJ as long as one end of it is within the window.
    window = dat[(dat["chr"] == chrom) &
               (
               ((dat["start"] >= new_start) & (dat["start"] <= new_end)) |
               ((dat["end"] >= new_start) & (dat["end"] <= new_end))
               )
                 ].copy()

    # dat["normalized_by_coverage"] = dat[
    #     "normalized_by_coverage"].replace([np.inf], 0)

    # if chrom == "chr22" and start == 33337801 and end == 33919994:
    #     print("herehsdyr")
    #     print(window)
    #     print(np.mean(window["uniquely_mapped"]))
    #     print(uniquely_mapped)

    # No adjacent SJs within a 10kb window. High chance this is noise.
    if window.shape[0] == 0:
        return 1e5

    return abs(np.log2(uniquely_mapped / np.mean(window["uniquely_mapped"])))


def get_percentile(numbers, percentile_threshold):
    return np.percentile(numbers, percentile_threshold, method="higher")


def get_df_with_rare_dna_variants(vcf_filepath, rare_af_threshold):
    mt = hl.import_vcf(vcf_filepath,
                       reference_genome="GRCh38",
                       force_bgz=True,
                       array_elements_required=False)

    ht = mt.rows()
    print("Filtering to rare variants...")
    ht = ht.filter(
                        (ht.info.gnomad_genomes_AF <= rare_af_threshold) |
                        (ht.info.cohort_AF <= rare_af_threshold)
                   )
    print(ht.count())
    ht = ht.select(ht.info.gnomad_genomes_AF,
                   ht.info.cohort_AF)
    df = ht.to_pandas()
    print(df)

    df["locus"] = df["locus"].astype(str)
    new_columns = ["chr", "pos"]
    df[new_columns] = df["locus"].str.split(":", expand=True)
    rare_variant_loci = df[["chr", "pos", "alleles", "info.splice_ai"]]
    rare_variant_loci.columns = ["chr", "pos", "alleles", "splice_ai"]
    rare_variant_loci["pos"] = rare_variant_loci["pos"].astype(int)

    return rare_variant_loci


def filter_by_rare_dna_variants(rare_variant_loci, chrom, start, end):
    window_size = 10000
    new_start = start - window_size
    new_end = end + window_size

    rare_variant_loci = rare_variant_loci[(rare_variant_loci["chr"] == chrom) &
                                          (rare_variant_loci["pos"] >= new_start) &
                                          (rare_variant_loci["pos"] <= new_end)]

    # TODO: add spliceAI score filtering.
    if rare_variant_loci.shape[0] > 0:
        # splice_ai_null_loci = rare_variant_loci[rare_variant_loci["splice_ai"].isna()]
        # for index, row in splice_ai_null_loci.iterrows():
        #     variant = f"{row['chr']}-{row['pos']}-{row['alleles'][0]}-{row['alleles'][1]}"
        #     if filter_by_spliceai_score(variant):
        #         return True
        #
        # splice_ai_not_null_loci = rare_variant_loci[rare_variant_loci["splice_ai"].notna()]
        # splice_ai_not_null_loci = splice_ai_not_null_loci[splice_ai_not_null_loci[
        #     "splice_ai"] >= 0.2]
        # if splice_ai_not_null_loci.shape[0] > 0:
        return True

    return False


def filter_by_spliceai_score(variant, distance=1000, score_threshold=0.2):
    url = f"https://spliceai-38-xwkwwwxdwq-uc.a.run.app/spliceai/?hg=38&&bc" \
          f"=comprehensive&variant" \
          f"={variant}&distance={distance}"
    response = requests.get(url)
    data = response.json()

    if "error" not in data: # If there are scores returned.
        scores = data["scores"]
        for score in scores:
            acceptor_gain = float(score["DS_AG"])
            acceptor_loss = float(score["DS_AL"])
            donor_gain = float(score["DS_DG"])
            donor_loss = float(score["DS_DL"])
            if (acceptor_gain >= score_threshold) or (acceptor_loss >= score_threshold) or \
                    (donor_gain >= score_threshold) or (donor_loss >= score_threshold):
                return True

    return False


def compare_with_gtex(gtex_normalized_dat, sample_normalized_dat, lr_coverage_threshold,
                      cur_sample):
    # in_rd_not_in_gtex_dat = get_sj_only_in_first_dat(dat, gtex_dat)
    print("Get SJ only in first dat...")
    in_rd_not_in_gtex_normalized_dat = get_sj_only_in_first_dat(sample_normalized_dat,
                                                                gtex_normalized_dat,
                                                                False)
    print("Only in RD")
    pd.set_option('display.max_columns', None)
    print(is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]))

    # in_rd_not_in_gtex_normalized_dat.to_csv(
    #     f"{cur_sample}_in_rd_not_in_gtex_normalized_dat_1.tsv",
    #     sep="\t", index=False)
    # in_rd_not_in_gtex_normalized_dat = pd.read_csv(f"{cur_sample}_in_rd_not_in_gtex_normalized_dat_1.tsv", sep="\t")

    # in_gtex_normalized_not_in_rd_dat = get_sj_only_in_first_dat(gtex_normalized_dat,
    #                                                             dat,
    #                                                             True)
    # in_gtex_normalized_not_in_rd_dat.to_csv(
    #     f"{cur_sample}_in_gtex_normalized_not_in_rd_dat_1.tsv",
    #     sep="\t", index=False)

    # in_gtex_normalized_not_in_rd_dat = pd.read_csv(f"{cur_sample}_in_gtex_normalized_not_in_rd_dat_1.tsv", sep="\t")

    # print("Number of splice junctions only in the GTEx data: ",
    #       in_gtex_normalized_not_in_rd_dat.shape[0])
    print("Compare with adjacent sj...")
    # in_gtex_normalized_not_in_rd_dat["lr_coverage"] = \
    #     in_gtex_normalized_not_in_rd_dat.apply(lambda row:
    #                                            compare_with_adjacent_sj(row["chr"],
    #                                                                     row["start"],
    #                                                                     row["end"],
    #                                                                     gtex_normalized_dat,
    #                                                                     row[
    #                                                                         "normalized_by_coverage"],
    #                                                                     row["gene"]),
    #                                            axis=1)
    # in_gtex_normalized_not_in_rd_dat.to_csv(
    #     f"{cur_sample}_in_gtex_normalized_not_in_rd_dat_2.tsv",
    #     sep="\t", index=False)
    #
    # in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
    #     in_rd_not_in_gtex_normalized_dat["gene"] == "PYROXD1"]
    # percentile_10_by_gene = sample_normalized_dat.groupby("gene")[
    #     "uniquely_mapped"].apply(get_percentile, 10)
    # TODO: uncomment down here
    # print("Getting percentile 10...")
    # percentile_to_use = 20
    # percentile_output_filepath = f"{cur_sample}_in_rd_not_in_gtex_normalized_dat_percentile_{percentile_to_use}.tsv"
    # overwrite = True
    # if os.path.exists(percentile_output_filepath) and not overwrite:
    #     print(f"{percentile_output_filepath} already exists. Loading from file.")
    #     in_rd_not_in_gtex_normalized_dat = pd.read_csv(percentile_output_filepath)
    # else:
    #     in_rd_not_in_gtex_normalized_dat["percentile_interval"] = \
    #         in_rd_not_in_gtex_normalized_dat.apply(lambda row: get_interval_percentile(
    #             # TODO: for GTEx, use gtex_normalized_dat
    #             sample_normalized_dat,
    #             row["chr"],
    #             row["start"],
    #             row["end"],
    #             percentile=percentile_to_use,
    #             window_size=10000),
    #                                                axis=1)
    #
    #     in_rd_not_in_gtex_normalized_dat.to_csv(percentile_output_filepath,
    #                                             index=False)
    # # in_rd_not_in_gtex_normalized_dat["percentile_10"] = \
    # #     in_rd_not_in_gtex_normalized_dat["gene"].map(percentile_10_by_gene)
    #
    # print("Before filtering by 20th percentile")
    # print(in_rd_not_in_gtex_normalized_dat.shape)
    # print(in_rd_not_in_gtex_normalized_dat)
    # print(is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]))
    #
    # in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
    #     in_rd_not_in_gtex_normalized_dat["uniquely_mapped"] >=
    #     in_rd_not_in_gtex_normalized_dat["percentile_interval"]]
    # print("After filtering by 20th percentile")
    # pd.set_option('display.max_columns', None)
    # print(in_rd_not_in_gtex_normalized_dat.shape)
    # print(is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]))

    pd.set_option('display.max_columns', None)
    print(is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]))
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
        in_rd_not_in_gtex_normalized_dat["num_samples_with_this_junction"] <= 10]
    print("After filtering by num_samples_with_this_junction")
    print(in_rd_not_in_gtex_normalized_dat.shape)
    print(is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]))

    print("Getting lr coverage...")
    in_rd_not_in_gtex_normalized_dat["lr_coverage"] = \
        in_rd_not_in_gtex_normalized_dat.apply(lambda row:
                                               compare_with_adjacent_sj(
                                                   sample_normalized_dat,
                                                   row["chr"],
                                                   row["start"],
                                                   row["end"],
                                                   row["uniquely_mapped"],
                                                   row["gene"]),
                                               axis=1)
    # in_rd_not_in_gtex_normalized_dat.to_csv(
    #     f"{cur_sample}_in_rd_not_in_gtex_normalized_dat_2.tsv",
    #     sep="\t", index=False)

    # in_gtex_normalized_not_in_rd_dat = pd.read_csv(f"{cur_sample}_in_gtex_normalized_not_in_rd_dat_2.tsv", sep="\t")
    # in_rd_not_in_gtex_normalized_dat = pd.read_csv(f"{cur_sample}_in_rd_not_in_gtex_normalized_dat_2.tsv", sep="\t")
    print(is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]))
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
        in_rd_not_in_gtex_normalized_dat["lr_coverage"] <= NOVEL_SJ_THRESHOLD]
    print(in_rd_not_in_gtex_normalized_dat.shape)

    # in_gtex_normalized_not_in_rd_dat = in_gtex_normalized_not_in_rd_dat[
    #     in_gtex_normalized_not_in_rd_dat["lr_coverage"] <= NOVEL_SJ_THRESHOLD]

    print("Filtering by DNA variants...")
    in_rd_not_in_gtex_normalized_dat["has_rare_dna_variant"] = \
        in_rd_not_in_gtex_normalized_dat.apply(lambda row: filter_by_rare_dna_variants(
            cur_rare_variant_loci, row["chr"], row["start"], row["end"]),
                                               axis=1)
    print(in_rd_not_in_gtex_normalized_dat["has_rare_dna_variant"])
    in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat[
        in_rd_not_in_gtex_normalized_dat["has_rare_dna_variant"] == True]

    print("Number of splice junctions only in the sample data after filtering: ",
          in_rd_not_in_gtex_normalized_dat.shape[0])
    is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample])
    print(in_rd_not_in_gtex_normalized_dat)

    if is_present_causal_sj(in_rd_not_in_gtex_normalized_dat, TRUTH_SET[cur_sample]):
        print("Causal SJ is captured.")
    else:
        print("Causal SJ is missed.")

    # in_gtex_not_in_rd_dat = get_sj_only_in_first_dat(gtex_dat, dat)
    # in_gtex_normalized_not_in_rd_dat = get_sj_only_in_first_dat(gtex_normalized_dat,
    #                                                             dat)
    # print("Compare between sample and GTEx...")
    # different_between_rd_and_gtex = get_sj_different_between_dats(dat,
    #                                                               gtex_normalized_dat,
    #                                                               lr_coverage_threshold)
    # different_between_rd_and_gtex.to_csv(
    #     f"{cur_sample}_different_between_rd_and_gtex.tsv", sep="\t", index=False)

    # different_between_rd_and_gtex = pd.read_csv(f"{cur_sample}_different_between_rd_and_gtex.tsv", sep="\t")

    # in_gtex_normalized_not_in_rd_dat = in_gtex_normalized_not_in_rd_dat.drop(
    #     columns=["lr_coverage"])
    # in_rd_not_in_gtex_normalized_dat = in_rd_not_in_gtex_normalized_dat.drop(
    #     columns=["lr_coverage"])
    # different_between_rd_and_gtex = different_between_rd_and_gtex.drop(
    #     columns=["lr_coverage"])
    print("in rd: ", in_rd_not_in_gtex_normalized_dat.shape[0])
    # print("in gtex: ", in_gtex_normalized_not_in_rd_dat.shape[0])
    # print("different: ", different_between_rd_and_gtex.shape[0])

    res = pd.concat([
        in_rd_not_in_gtex_normalized_dat,
        # in_gtex_normalized_not_in_rd_dat,
        # different_between_rd_and_gtex
    ])
    res.to_csv(f"{cur_sample}_transcriptome_wide_res.tsv", sep="\t", index=False)
    return res


def apply_filters(bed_dat, gtex_normalized_dat, sample_id):
    print("-----------------------------------")
    print("number here1")
    print(bed_dat.shape)

    # casual_sj = TRUTH_SET[sample_id]
    # is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)

    # 3. Compare reads and coverage with GTEx data.

    print("Normalize sample reads by mean exon coverage...")
    start_time = time.time()
    normalized_filtered_bed_dat = normalize_reads_by_gene(bed_dat,
                                                          # sample_gene_coverage_dict,
                                                          )
    end_time = time.time()
    print(f"Normalization took {end_time - start_time} seconds.")
    print(normalized_filtered_bed_dat)

    print("Done.")

    print("Normalize GTEx reads by mean exon coverage...")
    normalized_gtex_normalized_dat = normalize_reads_by_gene(
        gtex_normalized_dat,
        # gtex_gene_coverage_dict,
    )

    print("Compare sample metrics with GTEx metrics...")
    lr_coverage_threshold = 5
    filtered_bed_dat = compare_with_gtex(normalized_gtex_normalized_dat,
                                         normalized_filtered_bed_dat,
                                         lr_coverage_threshold,
                                         sample_id)

    print("Done.")
    print("number here2")
    print(filtered_bed_dat.shape)

    # # Close the BigWig file
    # bw.close()

    return filtered_bed_dat


def annotate_with_gene_names(junctions_bed_gz_path, sample_id):
    output_filepath = f"{sample_id}_annotated_junctions.bed"
    # if os.path.exists(output_filepath):
    #     print(f"{output_filepath} already exists. Skipping annotation.")
    #     return pd.read_csv(output_filepath, sep="\t")

    if sample_id == "gtex":
        bed = read_junctions_bed(junctions_bed_gz_path, USE_COLS)
        print("Raw:")
        print(bed.shape)

        # Keep splice junctions that have at least 2 uniquely-mapped reads
        # bed = bed[bed["uniquely_mapped"] >= UNIQUE_READS_THRESHOLD]
        # Keep splice junctions that have maximum spliced alignment overhang at least 20
        # bed = bed[bed["maximum_spliced_alignment_overhang"] >= 20]
        # print("After QC:")
        # print(bed.shape)
    else:
        bed = read_junctions_bed(junctions_bed_gz_path, SAMPLE_USE_COLS)
        print("Raw:")
        print(bed.shape)

        # Keep splice junctions that have at least 2 uniquely-mapped reads
        # bed = bed[bed["uniquely_mapped"] >= UNIQUE_READS_THRESHOLD]
        # Keep splice junctions that have maximum spliced alignment overhang at least 20
        # bed = bed[bed["maximum_spliced_alignment_overhang"] >= 20]
        # print("After QC:")
        # print(bed.shape)

        bed["interval"] = bed.apply(lambda x: get_genome_interval(x["chr"],
                                                                  x["start"],
                                                                  x["end"]),
                                    axis=1)
        # Read interval count from json file
        with open("num_sample_with_this_junction_count.json", "r") as f:
            interval_count = json.load(f)
        bed["num_samples_with_this_junction"] = bed["interval"].map(interval_count)
        # bed["num_samples_with_this_junction"] = bed[
        #     "num_samples_with_this_junction"].astype(int)
        bed["num_samples_total"] = MUSCLE_SAMPLE_TOTAL

    bed_basic = bed[["chr", "start", "end"]]
    bed_basic_filepath = f"./junctions_bed/{sample_id}_basic.bed"
    bed_basic.to_csv(bed_basic_filepath,
                     sep="\t",
                     header=False, index=False)

    os.system(f"bedtools intersect -a {bed_basic_filepath} -b "
              f"{gencode_v47_genes_bed_filepath} -wa -wb -loj | cut -d $'\t' -f 1,2,3,7 > "
              f"{output_filepath}")

    annotated_junctions = pd.read_csv(f"{output_filepath}", sep="\t", header=None)
    annotated_junctions.columns = ["chr", "start", "end", "gene"]

    # There are rows duplicated by row, start, and end because a splice junction
    # might overlap with multiple genes since strand information is not considered.
    # Strand is ignored because some splice junctions are not annotated with strand.
    bed_annotated = bed.merge(annotated_junctions,
                              on=["chr", "start", "end"],
                              how="left")

    bed_annotated.to_csv(output_filepath, sep="\t", index=False)
    return bed_annotated


def compute_average_exon_coverage_for_selected_genes(genes, bw):
    # TODO: think about how to handle splice junctions that lie in unannotated regions.
    # If the gene is unannotated, return None
    # Goals: 1. Flag such splice junctions 2. Do not affect the normalization of
    # other nearby splice junctions.
    gene_coverage_dict = {}
    # genes = genes[:1]
    # genes = ["TTN"]
    for cur_gene in genes:
        if cur_gene == ".":
            continue
        average_exon_coverage_for_cur_gene = compute_average_exon_coverage_for_a_gene(
            cur_gene, bw)
        if cur_gene not in gene_coverage_dict:
            gene_coverage_dict[cur_gene] = average_exon_coverage_for_cur_gene
    return gene_coverage_dict


def use_the_longest_exons(dat):
    dat = dat.sort_values(by="start", ascending=True)
    dat = dat.reset_index(drop=True)

    start_dict = {}  # Value: [index of the row, length of the exon]
    end_dict = {}  # Value: [index of the row, length of the exon]
    longest_exon_row_indices = set()
    for index, row in dat.iterrows():
        cur_start = row["start"]
        cur_end = row["end"]
        cur_exon_length = cur_end - cur_start + 1
        if cur_start in start_dict:
            if cur_exon_length < start_dict[cur_start][1]:
                continue
        start_dict[cur_start] = [index, cur_end - cur_start + 1]

        if cur_end in end_dict:
            if cur_exon_length < end_dict[cur_end][1]:
                continue
        end_dict[cur_end] = [index, cur_exon_length]

    for key in start_dict:
        longest_exon_row_indices.add(start_dict[key][0])
    for key in end_dict:
        longest_exon_row_indices.add(end_dict[key][0])

    longest_exons_in_cur_gene = dat.iloc[list(longest_exon_row_indices)]
    return longest_exons_in_cur_gene


def compute_average_exon_coverage_for_a_gene(gene, bw):
    gene_coords = gene_coords_dict[gene]
    gene_chr = gene_coords.split(":")[0]
    gene_start = int(gene_coords.split(":")[1].split("-")[0])
    gene_end = int(gene_coords.split(":")[1].split("-")[1])
    gene_coverage_in_intervals = bw.values(gene_chr, gene_start, gene_end)

    # Get the coverage of the gene.
    total_gene_coverage = 0
    total_bases = 0
    # For exons with the same start or end position, use the longest exon.
    # exons_in_cur_gene = gencode_v47_longest_exons[gencode_v47_longest_exons["gene"] == gene]
    exons_in_cur_gene = gencode_v47_longest_exons[
        (gencode_v47_longest_exons["chr"] == gene_chr) & \
        (gencode_v47_longest_exons["start"] >= gene_start) &
        (gencode_v47_longest_exons["end"] <= gene_end)]

    # Iterate over each exon and get the coverage.
    for _, row in exons_in_cur_gene.iterrows():
        exon_start = int(row["start"])
        exon_end = int(row["end"])
        adjusted_start = exon_start - gene_start + 1
        adjusted_end = exon_end - gene_start + 1
        exon_coverage_in_intervals = gene_coverage_in_intervals[
                                     adjusted_start:adjusted_end]
        total_gene_coverage += np.nansum(exon_coverage_in_intervals)
        total_bases += exon_end - exon_start + 1

    if total_bases == 0:
        return 1e-6
    return total_gene_coverage / total_bases


def get_coords_gene_dict(gene_coords_dict):
    coords_gene_start_dict = {}
    coords_gene_end_dict = {}
    for gene in gene_coords_dict:
        cur_coords = gene_coords_dict[gene]
        cur_chr = cur_coords.split(":")[0]
        cur_start = int(cur_coords.split(":")[1].split("-")[0])
        cur_end = int(cur_coords.split(":")[1].split("-")[1])
        if cur_chr not in coords_gene_start_dict:
            coords_gene_start_dict[cur_chr] = [[cur_start], [gene]]
        else:
            coords_gene_start_dict[cur_chr][0].append(cur_start)
            coords_gene_start_dict[cur_chr][1].append(gene)

        if cur_chr not in coords_gene_end_dict:
            coords_gene_end_dict[cur_chr] = [[cur_end], [gene]]
        else:
            coords_gene_end_dict[cur_chr][0].append(cur_end)
            coords_gene_end_dict[cur_chr][1].append(gene)
    return coords_gene_start_dict, coords_gene_end_dict


if __name__ == "__main__":
    hl.init(backend="spark",
            default_reference="GRCh38",
            gcs_requester_pays_configuration="cmg-analysis")

    gencode_v47_genes_bed_filepath = "gencode.v47.annotation.collapsed_only.gene.bed"
    gencode_v47_genes = pd.read_csv(gencode_v47_genes_bed_filepath, sep="\t",
                                    header=None)
    gene_coords_dict = get_gene_coords_dict(gencode_v47_genes)
    all_genes = set(gene_coords_dict.keys())
    coords_gene_start_dict, coords_gene_end_dict = get_coords_gene_dict(
        gene_coords_dict)

    # Note: the exon bed file contains exons with the same start or end position (
    # alternative exons). When computing average exon coverage, use the longest exon
    # to include all possible information.
    # gencode_v47_exons_bed_filepath = "gencode.v47.annotation.exon.longest.bed"
    gencode_v47_exons_bed_filepath = "MANE_exon_1based_start.tsv"
    gencode_v47_longest_exons = pd.read_csv(gencode_v47_exons_bed_filepath, sep="\t")
    print(gencode_v47_longest_exons)

    nums_of_SJ = {}
    captured = {}
    rdg_rna_seq_dat = read_from_airtable(RNA_SEQ_BASE_ID,
                                         DATA_PATHS_TABLE_ID,
                                         DATA_PATHS_VIEW_ID)
    rdg_rna_seq_dat = rdg_rna_seq_dat[
        rdg_rna_seq_dat["sample_id"].isin(TRUTH_SET.keys())]
    # junctions_bed_gz_paths = list(rdg_rna_seq_dat["junctions_bed"])
    # bigwig_paths = list(rdg_rna_seq_dat["coverage_bigwig"])
    # single_wgs_vcf_paths = list(rdg_rna_seq_dat["wgs_single_sample_vcf"])
    # single_wes_vcf_paths = list(rdg_rna_seq_dat["wes_single_sample_vcf"])
    # # print(single_sample_vcf_paths)
    # sample_ids = list(rdg_rna_seq_dat["sample_id"])
    # print(sample_ids)
    with open("sample_vcf_dict.json", "r") as f:
        sample_vcf_dict = json.load(f)

    for cur_sample in sample_vcf_dict:
        # if cur_sample != "41M_MW_M1":
        #     continue
        print(cur_sample)
        # TODO: uncomment here
        # if cur_sample in sample_vcf_dict:
        #     cur_single_vcf_path = sample_vcf_dict[cur_sample]
        # else:
        #     cur_row = rdg_rna_seq_dat[rdg_rna_seq_dat["sample_id"] == cur_sample]
        #     if not cur_row["wgs_single_sample_vcf"].isna().values[0]:
        #         cur_single_vcf_path = cur_row["wgs_single_sample_vcf"].values[0]
        #     elif not cur_row["wes_single_sample_vcf"].isna().values[0]:
        #         cur_single_vcf_path = cur_row["wes_single_sample_vcf"].values[0]
        #     else:
        #         print("No DNA variant file found for this sample.")
        #         continue
        cur_single_vcf_path = sample_vcf_dict[cur_sample]
        print(cur_single_vcf_path)
        cur_rare_variant_loci = get_df_with_rare_dna_variants(
            cur_single_vcf_path,
            rare_af_threshold=0.05)
        print(cur_rare_variant_loci)

        # TODO: annotate only once (save and check if exists)
        cur_junctions_bed_gz_path = rdg_rna_seq_dat[rdg_rna_seq_dat["sample_id"] ==
                                                    cur_sample]["junctions_bed"].values[0]
        cur_bigwig_path = rdg_rna_seq_dat[rdg_rna_seq_dat["sample_id"] ==
                                                    cur_sample]["coverage_bigwig"].values[0]

        sample_annotated = annotate_with_gene_names(cur_junctions_bed_gz_path,
                                                    cur_sample)
        # Keep splice junctions that have at least 10 uniquely-mapped reads
        sample_annotated = sample_annotated[sample_annotated["uniquely_mapped"] >= UNIQUE_READS_THRESHOLD]
        sample_annotated = sample_annotated[sample_annotated["maximum_spliced_alignment_overhang"] >= 20]

        gtex_annotated = annotate_with_gene_names(GTEx_NORMALIZED_BED, "gtex")
        # Keep splice junctions that have at least 2 uniquely-mapped reads
        gtex_annotated = gtex_annotated[
            gtex_annotated["uniquely_mapped"] >= 2]
        gtex_annotated = gtex_annotated[
            gtex_annotated["maximum_spliced_alignment_overhang"] >= 20]

        # Sort the table by chromosome, start, and end.
        sample_annotated = sample_annotated.sort_values(by=["chr", "start", "end"])
        gtex_annotated = gtex_annotated.sort_values(by=["chr", "start", "end"])
        # Move gene to the column following end.
        new_cols = ["chr", "start", "end", "gene", "uniquely_mapped", "strand",
                    "maximum_spliced_alignment_overhang",
                    "num_samples_with_this_junction", "num_samples_total"]
        sample_annotated = sample_annotated[new_cols]
        gtex_annotated = gtex_annotated[new_cols]

        # Get the coverage file.
        sample_bw = read_coverage_bigwig(cur_bigwig_path)
        gtex_bw = read_coverage_bigwig(GTEx_NORMALIZED_BIGWIG)

        # # Compute average exon coverage for each gene
        # genes = sample_annotated["gene"].unique()

        print(f"Computing average exon coverage for sample {cur_sample}...")
        # start = time.time()
        # gene_coverage_dict = compute_average_exon_coverage_for_selected_genes(all_genes, sample_bw)
        # with open(f"{cur_sample}_gene_coverage_dict.json", "w") as file:
        #     json.dump(gene_coverage_dict, file, indent=4)
        # end = time.time()
        # print(f"Computing average exon coverage took {end - start} seconds.")

        # with open(f"{cur_sample}_gene_coverage_dict.json", "r") as file:
        #     sample_gene_coverage_dict = json.load(file)

        print(f"Computing average exon coverage for GTEx...")
        start = time.time()
        # gtex_gene_coverage_dict = compute_average_exon_coverage_for_selected_genes(all_genes,
        #                                                                            gtex_bw)
        # with open("gtex_gene_coverage_dict.json", "w") as file:
        #     json.dump(gtex_gene_coverage_dict, file, indent=4)
        end = time.time()
        print(f"Computing average exon coverage took {end - start} seconds.")

        with open("gtex_gene_coverage_dict.json", "r") as file:
            gtex_gene_coverage_dict = json.load(file)

        # apply_filters
        filtered_bed_dat = apply_filters(sample_annotated,
                                         gtex_annotated,
                                         cur_sample, )
        # filtered_bed_dat = filtered_bed_dat[filtered_bed_dat["lr_coverage"] !=
        #                                np.inf]
        # filtered_bed_dat = filtered_bed_dat[filtered_bed_dat["lr_coverage"] !=
        #                                     0]
        # filtered_bed_dat = filtered_bed_dat[filtered_bed_dat["normalized_by_coverage"] !=
        #                                     np.inf]
        # filtered_bed_dat = filtered_bed_dat[filtered_bed_dat["normalized_by_coverage"] !=
        #                                     0]
        #
        print(filtered_bed_dat)
        casual_sj = TRUTH_SET[cur_sample]
        is_present = is_present_causal_sj(filtered_bed_dat, casual_sj)
        print(f"{cur_sample}: {casual_sj}")
        print(is_present)
        captured[cur_sample] = is_present
        nums_of_SJ[cur_sample] = filtered_bed_dat.shape[0]
        print(f"The total number of splice junctions after filtering is"
              f" {filtered_bed_dat.shape[0]}.")
        print("**************************************************")

    print(nums_of_SJ)
    print(len(nums_of_SJ))
    print(np.mean(list(nums_of_SJ.values())))
    print(captured)
