from tests import tests
from utils import log


if __name__ == "__main__":
    tests.test_generate_vibration_signal()
    tests.test_cavitation_effect()
    #tests.test_marine_condition_effect()
    log.log_info("All tests completed!")    