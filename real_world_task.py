"""
Sets up a local workspace with a real-world task.
"""

import shutil
from pathlib import Path

from tasks import Task
from config import BASE_DIR

_REAL_WORLD_WORKSPACE = BASE_DIR / "workspace" / "real_world"

def _setup_messy_json_task() -> tuple[Task, Path]:
    _REAL_WORLD_WORKSPACE.mkdir(parents=True, exist_ok=True)
    
    # Write a buggy JSON parser
    code = '''import json
import csv

def parse_logs(log_file, output_csv):
    """Parses a log file where each line is a JSON object.
    Some lines are corrupted or invalid JSON. We want to extract 'id' and 'status'
    and write them to a CSV.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    results = []
    for line in lines:
        data = json.loads(line)
        results.append([data['id'], data['status']])
        
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'status'])
        writer.writerows(results)
'''
    (_REAL_WORLD_WORKSPACE / "parser.py").write_text(code)
    
    # Write a test script
    test_code = '''import os
from parser import parse_logs

def test_parse_logs():
    # Setup test data
    with open('test_input.log', 'w') as f:
        f.write('{"id": 1, "status": "success"}\\n')
        f.write('{"id": 2, "status": "failed"}\\n')
        f.write('CORRUPTED LINE\\n')
        f.write('{"id": 4, "status": "pending"}\\n')
        
    try:
        parse_logs('test_input.log', 'output.csv')
    except Exception as e:
        print("TEST FAILED WITH EXCEPTION:")
        print(e)
        return False
        
    if not os.path.exists('output.csv'):
        print("TEST FAILED: output.csv not created")
        return False
        
    with open('output.csv', 'r') as f:
        content = f.read()
        
    expected = "id,status\\n1,success\\n2,failed\\n4,pending\\n"
    if content.replace('\\r\\n', '\\n') != expected:
        print("TEST FAILED: content mismatch")
        print("Expected:\\n" + expected)
        print("Got:\\n" + content)
        return False
        
    print("TEST PASSED")
    return True

if __name__ == "__main__":
    if test_parse_logs():
        exit(0)
    else:
        exit(1)
'''
    (_REAL_WORLD_WORKSPACE / "test_parser.py").write_text(test_code)
    
    description = """
We have a log parser in `parser.py` that reads a file containing JSON on each line and writes a CSV.
Currently, it crashes when it encounters corrupted or non-JSON lines.

Your task: Fix `parser.py` so that it safely ignores corrupted lines and continues parsing the rest of the file.

Use the `shell` tool to run `python test_parser.py` to verify your fix.
"""
    task = Task(name="real_json_parser", description=description.strip(), run_tests=None)
    return task, _REAL_WORLD_WORKSPACE

def get_real_world_task() -> tuple[Task, Path]:
    if _REAL_WORLD_WORKSPACE.exists():
        shutil.rmtree(_REAL_WORLD_WORKSPACE)
    return _setup_messy_json_task()
