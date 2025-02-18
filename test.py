import torch
import time

print(torch.__version__)


# matrices size
size = 10000

# define the size of matrices on CPU
matrix_cpu1 = torch.randn(size, size)
matrix_cpu2 = torch.randn(size, size)

# measure time on CPU
start_time = time.time()
result_cpu = torch.matmul(matrix_cpu1, matrix_cpu2)
end_time = time.time()
cpu_time = end_time - start_time
print(f"Time on CPU: {cpu_time:.4f} seconds.")

# define the size of matrices on GPU
matrix_gpu1 = matrix_cpu1.to("cuda")
matrix_gpu2 = matrix_cpu2.to("cuda")

# measure time on GPU
start_time = time.time()
result_gpu = torch.matmul(matrix_gpu1, matrix_gpu2)
end_time = time.time()
gpu_time = end_time - start_time
print(f"Time on GPU: {gpu_time:.4f} seconds.")

# compare results
print(f"Speed-up: {cpu_time/gpu_time}")
