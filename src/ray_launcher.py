import subprocess

print("Launching distributed agent experiment...")
subprocess.run(["python", "distributed_train.py"])
subprocess.run(["python", "agent_simulation.py"])
