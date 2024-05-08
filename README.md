The purpose of the project is to define a set of heuristics rules to capture causal splice junctions from the muscle trueth set.

Curent rules:
1. Keep splice junctions that have at least 1 uniquely-mapped reads.
2. Keep splice junctions that fall within the regions of these known disease genes: PYROXD1, NEB, RYR1, TTN, COL6A3, POMGNT1, DMD, COL6A1, CAPN3, LARGE1, SRPK3, MTM1, LAMA2, COL6A2

Potential rules:
1. Keep splice junctions that only occur in rare disease cohort but not in GTEx
2. Keep splice junctions that only occur in GTEx but not in rare disease cohort
3. Keep splice junctions that only occur in a certain number of samples in rare disease cohort
4. Keep splice junctions based on the ratio of reads/(acceptor + donor alternative reads) - similar to FRASER2 rule
