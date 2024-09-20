import psutil

logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

print(f"Logical cores: {logical_cores}")
print(f"Physical cores: {physical_cores}")
