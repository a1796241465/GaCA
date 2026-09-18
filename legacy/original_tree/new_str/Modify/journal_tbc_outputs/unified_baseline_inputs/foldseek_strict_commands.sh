
TRAIN_STRUCTURES=/home/lihaotian/new_ec/train_structures
TEST_LT30_STRUCTURES=/home/lihaotian/new_ec/test_structures
TEST_3050_STRUCTURES=/home/lihaotian/new_ec/test_30-50_structures
OUT_DIR=/home/lihaotian/new_ec/foldseek_strict_outputs
mkdir -p ${OUT_DIR}

foldseek easy-search ${TEST_LT30_STRUCTURES} ${TRAIN_STRUCTURES} ${OUT_DIR}/foldseek_lt30.tsv ${OUT_DIR}/tmp_lt30 --format-output "query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore"

foldseek easy-search ${TEST_3050_STRUCTURES} ${TRAIN_STRUCTURES} ${OUT_DIR}/foldseek_30_50.tsv ${OUT_DIR}/tmp_30_50 --format-output "query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore"

