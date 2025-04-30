import os
import hail as hl
import argparse
import logging

num_cpu = 4
REGION = ["us-central1"]
# ANNOTATION_HT = "gs://seqr-reference-data/v03/GRCh38/SNV_INDEL/reference_datasets/combined.ht"
# ANNOTATION_HT is outdated
# gnomad_genomes = "gs://gcp-public-data--gnomad/release/4.1/ht/genomes/gnomad.genomes.v4.1.sites.ht"
# gnomad_exomes = "gs://gcp-public-data--gnomad/release/4.1/ht/exomes/gnomad.exomes.v4.1.sites.ht"
gnomad_joint = "gs://gcp-public-data--gnomad/release/4.1/ht/joint/gnomad.joint.v4.1.sites.ht"

logging.basicConfig(level=logging.INFO)


def convert_str_to_float(attribute):
    if attribute.dtype == hl.tfloat32 or attribute.dtype == hl.tfloat64:
        return attribute
    return hl.case().when(attribute == "nul", float('nan')).default(hl.float(attribute))


def add_annotation(mt, annotation_ht):
    key_mt = annotation_ht[mt.row_key]
    key_mt.describe()
    if "cohort_AC" in mt.info:
        mt = mt.annotate_rows(info=hl.struct(
            cohort_AC=mt.info.cohort_AC,
            cohort_AF=mt.info.cohort_AF,
            cohort_AN=mt.info.cohort_AN,
            gnomad_exomes_AF=convert_str_to_float(
                key_mt.exomes.fafmax.faf95_max
            ),
            gnomad_genomes_AF=convert_str_to_float(
                key_mt.genomes.fafmax.faf95_max
            ),
        ))
    else:
        mt = mt.annotate_rows(info=hl.struct(
            cohort_AC=hl.int(mt.info.AC[mt.a_index - 1]),
            cohort_AF=convert_str_to_float(mt.info.AF[mt.a_index - 1]),
            cohort_AN=hl.int(mt.info.AN),
            # clinvar_allele_id=key_mt.clinvar.alleleId,
            # #     clinvar_pathogenicity_id = key_mt.clinvar.pathogenicity_id,
            # clinvar_clinsig=mt.enums.clinvar.pathogenicity[
            #     key_mt.clinvar.pathogenicity_id],
            # clinvar_gold_stars=key_mt.clinvar.goldStars,
            # CADD=convert_str_to_float(key_mt.cadd.PHRED),
            # eigen=convert_str_to_float(key_mt.eigen.Eigen_phred),
            # revel=convert_str_to_float(
            #     key_mt.dbnsfp.REVEL_score),
            # # splice_ai=convert_str_to_float(key_mt.splice_ai.delta_score),
            # primate_ai=convert_str_to_float(key_mt.primate_ai.score),
            # exac_AF=convert_str_to_float(key_mt.exac.AF_POPMAX),
            # gnomad_exomes_AF=convert_str_to_float(
            #     key_mt.genomes.AF_POPMAX_OR_GLOBAL),
            # gnomad_genomes_AF=convert_str_to_float(
            #     key_mt.gnomad_genomes.AF_POPMAX_OR_GLOBAL),
            # topmed_AF=convert_str_to_float(key_mt.topmed.AF),
            gnomad_exomes_AF=convert_str_to_float(
                key_mt.exomes.fafmax.faf95_max
            ),
            gnomad_genomes_AF=convert_str_to_float(
                key_mt.genomes.fafmax.faf95_max
            ),
        ))
    return mt


def main():
    hl.init(default_reference="GRCh38")
    single_sample_vcf_path = args.input_vcf
    logging.info(f"The single sample vcf to use is {single_sample_vcf_path}")
    mt = hl.import_vcf(single_sample_vcf_path,
                       min_partitions=2000,
                       reference_genome="GRCh38",
                       force_bgz=True,
                       array_elements_required=False)
    mt.describe()
    # if "gnomad_genomes_AF" in mt.info and "splice_ai" in mt.info:
    #     # if mt.info.gnomad_genomes_AF.dtype == hl.tfloat32 or mt.info.gnomad_genomes_AF.dtype == hl.tfloat64:
    #     os.system(f"gcloud storage cp {args.input_vcf} {args.output_vcf}")
    #     return

    # mt = mt.select_entries(mt.AD, mt.AF, mt.GT, mt.DP, mt.GP, mt.GQ)
    mt = mt.select_entries(mt.AD, mt.GT, mt.DP, mt.GQ)
    mt = mt.drop(mt.filters, mt.qual)  # The FILTER field contains some unlabeld
    # filters that causes errors in spliceAI; QUAL scores are not available.
    mt.describe()

    filtered_mt = mt.filter_rows(
        hl.agg.any(mt.GT.is_non_ref() & (mt.DP > 0))
    )  # in AD, 0 is the reference allele;  1 is the alternate allele.
    filtered_mt.rows().show()
    filtered_split_mt = hl.split_multi_hts(filtered_mt)

    # annotation_ht = hl.read_table(args.annotation_ht)
    # genomes_ht = hl.read_table(gnomad_genomes)
    # exomes_ht = hl.read_table(gnomad_exomes)
    joint_ht = hl.read_table(gnomad_joint)

    print("Adding additional annotation...")
    filtered_split_mt = filtered_split_mt.annotate_globals(
        **joint_ht.index_globals())
    filtered_split_mt = add_annotation(filtered_split_mt,
                                       joint_ht)

    output_vcf_path = args.output_vcf
    # Handle haploid genotypes.
    filtered_split_mt = filtered_split_mt.annotate_entries(
        GT=hl.if_else(
            filtered_split_mt.GT.ploidy == 1,
            hl.call(filtered_split_mt.GT[0], filtered_split_mt.GT[0]),
            filtered_split_mt.GT)
    )
    print("Writing to single sample vcf...")
    hl.export_vcf(filtered_split_mt, output_vcf_path, tabix=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-vcf",
        type=str,
        help="Google Cloud path to retrieve input VCF file.",
        required=True,
    )
    parser.add_argument(
        "--output-vcf",
        type=str,
        help="Google Cloud path to write output VCF file to.",
        required=True,
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        help="ID of the sample.",
        required=True,
    )
    # parser.add_argument(
    #     "--annotation-ht",
    #     type=str,
    #     help="Google Cloud path to retrieve annotation HT file.",
    #     default=ANNOTATION_HT,
    # )
    args = parser.parse_args()

    main()
