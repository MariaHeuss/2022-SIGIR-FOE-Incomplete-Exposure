from tests.test_bvn_decomposition import run_bvn_tests
from tests.test_metrics import run_metrics_tests
from tests.test_run_foeir import test_individual_fairness_constraint

run_bvn_tests()
run_metrics_tests()
test_individual_fairness_constraint()
