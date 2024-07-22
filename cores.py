import psutil

# Get the number of logical cores
logical_cores = psutil.cpu_count(logical=True)

# Get the number of physical cores
physical_cores = psutil.cpu_count(logical=False)

print(f"Logical cores: {logical_cores}")
print(f"Physical cores: {physical_cores}")
