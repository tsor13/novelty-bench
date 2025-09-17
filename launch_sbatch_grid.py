#!/usr/bin/env python3
"""
Generic grid launcher using sbatch for SLURM job submission.

➤ Submit all combos as separate sbatch jobs:
    python launch_sbatch_grid.py sweep.yaml

➤ Submit one combo only (for testing):
    python launch_sbatch_grid.py sweep.yaml --index 0

➤ Print sbatch commands without submitting:
    python launch_sbatch_grid.py sweep.yaml --print

➤ Limit concurrent jobs (good cluster citizenship):
    python launch_sbatch_grid.py sweep.yaml --max-concurrent 5
"""
import argparse
import itertools
import shlex
import subprocess
import sys
import time

import yaml


# ------------- helpers -------------------------------------------------
def normalize_env_key(key):
    """Convert config keys to ENV_VAR style."""
    return key.upper().replace('-', '_')


def section_to_env_and_args(section):
    """Extract environment variables and extra CLI args from a section."""
    env = {}
    extra_args = []

    if section is None:
        return env, extra_args

    if isinstance(section, dict):
        explicit_env = section.get('env', {})
        for key, value in explicit_env.items():
            env[normalize_env_key(key)] = str(value)

        for key, value in section.items():
            if key in {'env', 'extra_flags', 'extra_args'}:
                continue
            env[normalize_env_key(key)] = str(value)

        extra_args.extend(str(flag) for flag in section.get('extra_flags', []))
        extra_args.extend(str(arg) for arg in section.get('extra_args', []))

    return env, extra_args


def build_commands(cfg, index=None):
    if 'command' not in cfg or not cfg['command']:
        sys.exit("ERROR: 'command' must be provided in the YAML config!")

    cmd_prefix = shlex.split(cfg['command'])

    combos = list(itertools.product(
        cfg.get('runs', [{}]),
        cfg.get('datasets', [{}])
    ))

    if index is not None:
        try:
            combos = [combos[index]]
        except IndexError as exc:
            raise IndexError from exc

    commands = []
    for run, dset in combos:
        env = {}
        args = list(cmd_prefix)

        for section in (cfg.get('base', {}), dset, run):
            section_env, section_args = section_to_env_and_args(section)
            env.update(section_env)
            args.extend(section_args)

        commands.append({'command': args, 'env': env})

    return commands


def build_wrap_command(command_args, env):
    """Create a shell snippet that sets env vars then runs the command."""
    env_assignments = ' '.join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())
    )
    command_str = ' '.join(shlex.quote(arg) for arg in command_args)

    if env_assignments:
        return f"{env_assignments} {command_str}".strip()
    return command_str


def build_sbatch_command(job_cmd, env, sbatch_config, job_index):
    """Build sbatch command with the given job command and env."""
    sbatch_cmd = ['sbatch']

    for key, value in sbatch_config.items():
        if key in {'wrap_command', 'job-name'}:
            continue
        if isinstance(value, bool):
            if value:
                sbatch_cmd.append(f"--{key}")
        else:
            sbatch_cmd.extend([f"--{key}", str(value)])

    job_name_base = sbatch_config.get('job-name', 'bb-eval')
    sbatch_cmd.extend(['--job-name', f"{job_name_base}-{job_index}"])

    wrap_body = build_wrap_command(job_cmd, env)
    wrap_template = sbatch_config.get('wrap_command')
    if wrap_template:
        wrap_body = wrap_template.format(command=wrap_body)

    sbatch_cmd.extend(['--wrap', wrap_body])
    return sbatch_cmd


def get_running_jobs(job_name_prefix):
    """Get count of currently running/pending jobs with given name prefix."""
    try:
        result = subprocess.run(
            ['squeue', '-u', subprocess.getoutput('whoami'), '-h', '-o', '%j'],
            capture_output=True, text=True, check=True
        )
        job_names = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return sum(1 for name in job_names if name.startswith(job_name_prefix))
    except subprocess.CalledProcessError:
        return 0


def wait_for_job_slots(job_name_prefix, max_concurrent):
    """Wait until there are fewer than max_concurrent jobs running."""
    while True:
        running = get_running_jobs(job_name_prefix)
        if running < max_concurrent:
            break
        print(f"⏳ Waiting... {running}/{max_concurrent} jobs running with prefix '{job_name_prefix}'")
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_cfg')
    parser.add_argument('--index', type=int,
                        help='Submit only combo at this flat index')
    parser.add_argument('--print', action='store_true',
                        help='Print sbatch commands without submitting them')
    parser.add_argument('--max-concurrent', type=int,
                        help='Maximum number of concurrent jobs to allow')
    args = parser.parse_args()

    with open(args.yaml_cfg, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    if 'sbatch' not in cfg:
        sys.exit("ERROR: No 'sbatch' section found in YAML config!")

    try:
        jobs = build_commands(cfg, args.index)
    except IndexError:
        sys.exit(f"Index {args.index} out of range!")

    failed_submissions = []
    submitted_jobs = []
    job_name_base = cfg['sbatch'].get('job-name', 'bb-eval')

    for idx, job in enumerate(jobs):
        if args.max_concurrent and not args.print:
            wait_for_job_slots(job_name_base, args.max_concurrent)

        sbatch_cmd = build_sbatch_command(job['command'], job['env'], cfg['sbatch'], idx)

        env_preview = ' '.join(f"{k}={v}" for k, v in sorted(job['env'].items())) or '(none)'
        print(f"\n▶ [{idx + 1}/{len(jobs)}] Submitting job:")
        print(f"Env: {env_preview}")
        print(f"Command: {' '.join(job['command'])}")
        print(f"Sbatch command: {' '.join(sbatch_cmd)}")

        if args.print:
            continue

        try:
            result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1]
            submitted_jobs.append((idx + 1, job_id))
            print(f"✅ Job submitted: {job_id}")
        except subprocess.CalledProcessError as exc:
            print(f"❌ Sbatch submission failed with exit code {exc.returncode}")
            print(f"Error output: {exc.stderr}")
            failed_submissions.append((idx + 1, sbatch_cmd))
            print('Continuing to next job...\n')
            continue

    if args.print:
        return

    if submitted_jobs:
        print(f"\n✅ Successfully submitted {len(submitted_jobs)} job(s):")
        for job_num, job_id in submitted_jobs:
            print(f"  [{job_num}] Job ID: {job_id}")

    if failed_submissions:
        print(f"\n⚠️  {len(failed_submissions)} job submission(s) failed:")
        for job_num, cmd in failed_submissions:
            print(f"  [{job_num}] {' '.join(cmd)}")


if __name__ == '__main__':
    main()
