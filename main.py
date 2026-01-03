from tests import tests
from utils import log


if __name__ == "__main__":
    tests.test_generate_vibration_signal()
    tests.test_cavitation_effect()
    feature_matrix, feature_names = tests.test_feature_extraction()
    tests.test_generate_dataset()
    log.log_info("finish test!")    