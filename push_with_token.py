#!/usr/bin/env python3
"""
Push to GitHub using token from .env
"""
import subprocess
import os

# Read token from .env
with open('.env', 'r') as f:
    token = f.read().strip()

print(f"[*] Token loaded: {token[:20]}...{token[-10:]}")

# Configure git remote with token
remote_url = f"https://{token}@github.com/chekrireddy/Insurance-Fraud-Detection-ML.git"
cmd_set_remote = ["git", "remote", "set-url", "origin", remote_url]

print("\n[*] Configuring git remote...")
result = subprocess.run(cmd_set_remote, capture_output=True, text=True)
if result.returncode == 0:
    print("[OK] Remote configured")
else:
    print(f"[ERROR] Failed to configure remote: {result.stderr}")
    exit(1)

# Push to GitHub
print("\n[*] Pushing to GitHub...")
cmd_push = ["git", "push", "origin", "master", "-v"]
result = subprocess.run(cmd_push, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode == 0:
    print("\n[SUCCESS] Code pushed to GitHub!")
    print("Repository: https://github.com/chekrireddy/Insurance-Fraud-Detection-ML")
else:
    print(f"\n[ERROR] Push failed with code {result.returncode}")

# Reset remote URL (remove token from config)
print("\n[*] Resetting remote URL (removing token from config)...")
cmd_reset = ["git", "remote", "set-url", "origin", "https://github.com/chekrireddy/Insurance-Fraud-Detection-ML.git"]
subprocess.run(cmd_reset, capture_output=True)
print("[OK] Remote URL reset")

exit(result.returncode)
