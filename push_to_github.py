#!/usr/bin/env python3
"""
Git push helper script
"""
import subprocess
import sys

def run_cmd(cmd):
    """Run command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr, result.returncode

# Check status
print("Checking git status...")
status, _ = run_cmd("git status")
print(status)

print("\nChecking remote...")
remote, _ = run_cmd("git remote -v")
print(remote)

print("\nChecking commits...")
log, _ = run_cmd("git log --oneline -3")
print(log)

print("\nAttempting push...")
push_output, push_code = run_cmd("git push -u origin master --verbose 2>&1")
print(push_output)

if push_code == 0:
    print("\n[SUCCESS] Push completed!")
else:
    print(f"\n[ERROR] Push failed with code {push_code}")
    print("You may need to authenticate with GitHub")
    print("\nTry running: git push origin master")

sys.exit(push_code)
