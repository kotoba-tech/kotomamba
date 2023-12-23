# Mamba

## Installation

1. install PyTorch

    ```bash
    python -m venv .venv
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

2. module load

    **GPU** is required to install `causal-conv1d`, `mamba-ssm`.
    so, interactive job may be required.

    If you are using ABCI, you can load the following modules.
    ```bash
    source /etc/profile.d/modules.sh
    module load cuda/11.8/11.8.0
    module load cudnn/8.9/8.9.2
    module load nccl/2.16/2.16.2-1
    module load hpcx/2.12
    ```

3. install Mamba

    ```bash
    pip install causal-conv1d>=1.1.0
    pip install mamba-ssm
    ```

4. change triton code

    ldconfig path is not recognized correctly, change to full path


    `.venv/lib/python3.10/site-packages/triton/common/build.py:L21`
    ```diff
    @functools.lru_cache()
    def libcuda_dirs():
    -    libs = subprocess.check_output(["ldconfig", "-p"]).decode()
    +    libs = subprocess.check_output(["/usr/sbin/ldconfig", "-p"]).decode()
    ```

## Inference

you can measure throughput of inference in 2.8B size model
```bash
qsub -g $GROUP scripts/abci/inference/throughput_test.sh
```

