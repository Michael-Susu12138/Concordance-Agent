import subprocess
import time

#list of services in the format ("module:app", port)
services = [
    ("mcp_server:app", 8000),
    ("radiologist_agent:app", 8001),
    ("pathologist_agent:app", 8002),
    ("concordance_agent:app", 8003),
    ("notification_agent:app", 8004),
    ("coordinator_agent:app", 8005),
]

processes = []
for module_app, port in services:
    cmd = [
        "uvicorn",
        module_app,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    print(f"Starting {module_app} on port {port}...")
    p = subprocess.Popen(cmd)
    processes.append(p)
    time.sleep(0.5) # small delay to stagger startup

print("All services started. Press Ctrl+C to stop.")
try:
    # wait for all processes to complete (or be terminated)
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("Shutting down services...")
    for p in processes:
        p.terminate()

# (how to run) in terminal:
# first make sure 'Agents' under home directory, if not, use cd command
# enter: python .\run_services.py
# enter: curl.exe http://localhost:8000/mcp/patient/3085 (check whether the mcp server works normally)
# enter: curl.exe -X POST http://localhost:8005/process_case/3085