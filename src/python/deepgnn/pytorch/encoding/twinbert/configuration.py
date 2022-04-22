class DeepSpeedArgs:
    def __init__(self, config: dict, local_rank=0):

        dsconfig = config["deepspeed"] if "deepspeed" in config else {}

        # Use DeepSpeed transformer kernel to accelerate.
        self.deepspeed_transformer_kernel = (
            dsconfig["transformer_kernel"]
            if "transformer_kernel" in dsconfig
            else False
        )

        # Total batch size for training, only used for summary writer.
        self.train_batch_size = (
            dsconfig["train_batch_size"] if "train_batch_size" in dsconfig else 1024
        )

        self.train_micro_batch_size_per_gpu = (
            dsconfig["train_micro_batch_size_per_gpu"]
            if "train_micro_batch_size_per_gpu" in dsconfig
            else 1024
        )

        # Use stochastic mode for high-performance transformer kernel.
        self.stochastic_mode = (
            dsconfig["stochastic_mode"] if "stochastic_mode" in dsconfig else False
        )

        # Use DeepSpeed transformer kernel memory optimization to perform invertible normalize
        # backpropagation.
        self.normalize_invertible = (
            dsconfig["normalize_invertible"]
            if "normalize_invertible" in dsconfig
            else False
        )

        # random seed for initialization
        self.seed = dsconfig["seed"] if "seed" in dsconfig else 42

        self.local_rank = local_rank

        # use global fp16 setting
        self.fp16_enabled = config["enable_fp16"] if "enable_fp16" in config else False

        self.apex = dsconfig["apex"] if "apex" in dsconfig else True
