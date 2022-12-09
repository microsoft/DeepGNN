import ray
from time import sleep

def func1(i: int) -> int:
    sleep(1)
    return i + 1

# Create a dataset and then create a pipeline from it.
base = ray.data.range(1000)
print(base)
# -> Dataset(num_blocks=200, num_rows=1000000, schema=<class 'int'>)
pipe = base.window(blocks_per_window=10)
print(pipe)
# -> DatasetPipeline(num_windows=20, num_stages=1)

# Applying transforms to pipelines adds more pipeline stages.
pipe = pipe.map(func1)
print(pipe)
# -> DatasetPipeline(num_windows=20, num_stages=4)

# Output can be pulled from the pipeline concurrently with its execution.
num_rows = 0
for row in pipe.iter_torch_batches(batch_size=5):#.iter_rows():
    num_rows += 1
    print(num_rows)
# ->
# Stage 0:  55%|█████████████████████████                |11/20 [00:02<00:00,  9.86it/s]
# Stage 1:  50%|██████████████████████                   |10/20 [00:02<00:01,  9.45it/s]
# Stage 2:  45%|███████████████████                      | 9/20 [00:02<00:01,  8.27it/s]
# Stage 3:  35%|████████████████                         | 8/20 [00:02<00:02,  5.33it/s]
print("Total num rows", num_rows)
# -> Total num rows 1000000
