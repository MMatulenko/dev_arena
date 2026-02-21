"""
Sets up and manages a SWE-bench task using Docker.
"""

import json
import logging
import subprocess
from pathlib import Path
from datasets import load_dataset

from tasks import Task
from docker_tools import DockerTools

logger = logging.getLogger(__name__)

def run_swe_bench_evaluation(instance_id: str):
    # This just checks if the container is running and runs the test_patch
    pass

def setup_swe_bench_task(instance_id: str) -> tuple[Task, DockerTools, str]:
    """
    Pulls a pre-built SWE-bench Lite image and starts a container.
    """
    logger.info("Loading SWE-bench Lite dataset...")
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    
    # Find the task
    task_data = next((t for t in ds if t['instance_id'] == instance_id), None)
    if not task_data:
        raise ValueError(f"Task {instance_id} not found in SWE-bench Lite")
        
    problem_statement = task_data['problem_statement']
    repo = task_data['repo']

    # Parse test IDs for the CI Oracle
    try:
        fail_to_pass = json.loads(task_data.get('FAIL_TO_PASS', '[]'))
    except (json.JSONDecodeError, TypeError):
        fail_to_pass = []
    try:
        pass_to_pass = json.loads(task_data.get('PASS_TO_PASS', '[]'))
    except (json.JSONDecodeError, TypeError):
        pass_to_pass = []

    test_patch = task_data.get('test_patch', '')

    description = f"""
    You are working on a real GitHub issue for the repository `{repo}`.

    PROBLEM STATEMENT:
    {problem_statement}

    INSTRUCTIONS:
    1. The repository is mounted at your current working directory (`/testbed`).
    2. Use your tools to find the buggy code and write a patch.
    3. You can run tests using `shell('pytest <test_file>')` or similar commands.
    4. DO NOT write `test_patch` or modify tests unless explicitly asked to. Focus on fixing the source code to resolve the problem.
    5. Once you are confident the issue is fixed, output "VERIFICATION: PASSED".
    """
    task = Task(name=instance_id, description=description.strip(), run_tests=None)
    
    image_name = f"huyouare/swebench-lite:sweb.eval.x86_64.{instance_id}"
    container_name = f"swe_bench_{instance_id}"
    
    logger.info("Ensuring Docker image %s is available...", image_name)
    res = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True)
    if not res.stdout.strip():
        logger.info("Pulling image (this might take a few minutes)...")
        subprocess.run(["docker", "pull", image_name], check=True)
        
    logger.info("Starting container %s...", container_name)
    # Remove existing container if any
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    
    # Start container in detached mode, tailing dev/null to keep it alive
    subprocess.run([
        "docker", "run", "-d", "--name", container_name,
        "-w", "/testbed", image_name, "tail", "-f", "/dev/null"
    ], check=True)
    
    logger.info("CI Oracle: %d FAIL_TO_PASS tests, %d PASS_TO_PASS tests", len(fail_to_pass), len(pass_to_pass))
    tools = DockerTools(
        container_name=container_name,
        workspace_dir="/testbed",
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
    )
    
    if test_patch:
        logger.info("Applying test_patch to container...")
        # Write test patch to container and apply it
        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(test_patch)
            temp_path = f.name
        
        subprocess.run(["docker", "cp", temp_path, f"{container_name}:/tmp/test_patch.diff"], capture_output=True)
        os.remove(temp_path)
        
        # Apply patch and commit so that git reset --hard doesn't wipe it out between sprints
        tools._exec("git apply -v /tmp/test_patch.diff || patch -p1 < /tmp/test_patch.diff")
        tools._exec("git config user.email 'test@example.com' && git config user.name 'Test' && git add . && git commit -m 'Apply SWE-bench test patch'")
        
    return task, tools, container_name

def teardown_swe_bench_task(container_name: str):
    logger.info("Tearing down container %s...", container_name)
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
