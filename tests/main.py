"""
Entry point for tests
cmd args: [sys.executable, '-m', 'pytest',
           '/tests', 'v', '--tbs=short', '--disable-warnings', f'--rootdir={project_root}',
           '--cov=src', '--cov-report=term-missing', '--slow', '-m', 'not slow']
"""
import argparse
import os
import subprocess
import sys
from utils import log

def run_tests():
    """
    Tests configurations for marine pump cavitation ML.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    cmd = [sys.executable, '-m', 'pytest', project_root, '-v', '--tb=short', '--disable-warnings', 
           f'--rootdir={project_root}',
        ]
    
    if '--coverage' in sys.argv:
        cmd.extend(['--cov=src', '--cov-report=term-missing', '--cov-report=html'])
    
    if '--slow' in sys.argv:
        cmd.append('-m')
        cmd.append('slow')
    else:
        cmd.append('-m')
        cmd.append('not slow')
    try:
        results = subprocess.run(cmd, check=False)
        if results.returncode == 0:
            pass
        else:
            return results.returncode
    except Exception as e:
        log.log_error(f'fail to run tests {e}')
        return 1
    

def run_specific_test(test_file, test_name=None):
    """
    Function for running the specific test 
    """
    cmd = [sys.executable, '-m', 'pytest', f'{test_file}', '-v', '--tb=short', '--disable-warnings']
    if test_name:
        cmd.append(f"-k{test_name}")
    if '--slow' in sys.argv:
        cmd.append('-m')
        cmd.append('slow')
    else:
        cmd.append('-m')
        cmd.append('not slow')
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return exit(1)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test runner for marine-pump-cavitation-ML")
    parser.add_argument('--file', action='store', type=str, help='File to run the test')
    parser.add_argument('--function', action='store', type=str, help='Run test for this particular function')
    parser.add_argument('--data', action='store_true', help='Run test for vibration data generator')
    parser.add_argument('--features', action='store_true', help='Run test for features extraction')
    parser.add_argument('--all', action='store_true', help='Run all tests [default]')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--slow', action='store_true', help='Including the slow tests')
    args = parser.parse_args()

    if args.file and args.function:
       sys.exit(run_specific_test(args.file, args.function))
    elif args.file:
       sys.exit(run_specific_test(args.file))
    elif args.data:
       sys.exit(run_specific_test('tests_vibration.py'))
    elif args.features:
       sys.exit(run_specific_test('tests_time_features.py'))
    else:
       sys.exit(run_tests())
    
    
     

