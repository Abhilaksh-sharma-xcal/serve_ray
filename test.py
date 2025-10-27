import ray, subprocess

@ray.remote
def check_nvcc():
    result = subprocess.run("which nvcc", shell=True, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()

ray.init(address="https://blitz-ray-dashboard.xcaliberapis.com/")
print(ray.get(check_nvcc.remote()))
