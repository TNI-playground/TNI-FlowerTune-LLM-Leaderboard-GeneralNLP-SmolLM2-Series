# FlowerTune LLM on General NLP Dataset

This directory conducts federated instruction tuning with pretrained SmolLM2 Series Models: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) on a [General NLP dataset](https://huggingface.co/datasets/vicgalle/alpaca-gpt4).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.


## Methodology

This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedAvg strategy.
This provides a baseline performance for the leaderboard of General NLP challenge.


## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
conda create -n flower python=3.10
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `100` rounds.
All settings are defined in `pyproject.toml`.

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).


## Running the challenge

Follow the instruction [here](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) to log in your account. Note you only need to complete this stage once in your development machine:

```bash
huggingface-cli login
```

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

### Benchmark

All the experiments are conducted on a NVIDIA GeForce GTX 1080 (8 GB).

| Challenges                      | STEM       |   Social   |  Humanities |   Avg      |
| :--------:                      | :--------: | :--------: | :--------:  | :--------: |
|[SmolLM2-135M-Instruct](https://drive.google.com/drive/folders/1x7J2tosTtlkvUXSJGyB-ZHxUZhU3cMM4?usp=sharing) (100Rounds)| 18.68      |  21.90     |  22.74      |   21.10    |
|[SmolLM2-135M](https://drive.google.com/drive/folders/1oPrFSA3URA7u9D0t7Tw7ztmOztn6CcdY?usp=sharing) (100Rounds)         | 2.94       |  3.31      |  2.29       |   2.84     |
|[SmolLM2-360M-Instruct](https://drive.google.com/drive/folders/1395vuT_HVPHEdrtlSk4oU3m2c-mflWGq?usp=sharing) (100Rounds)| 19.44      |  19.43     |  12.68      |   17.18    |

SmolLM2-360M performed poorly under the same hyperparameters, so its results are omitted here.

## VRAM consumption

We use models with 4-bit quantization as default. The estimated VRAM consumption per client for each challenge is shown below:

|Models  |SmolLM2-135M-Instruct (BS=16)|SmolLM2-360M-Instruct (BS=8)|SmolLM2-135M (BS=16)|SmolLM2-360M (BS=8,Round=300)|
| :----: | :--------:                  | :--------:                 | :--------:         | :--------:         |
|VRAM    |        8.03 GB              |    7.93 GB                 |    8.07 GB         |      7.80 GB       |
|Comm    |        747.07 MB            |    1321.88 MB              |    747.07 MB       |      3965.62 MB           | 

You can adjust the CPU/GPU resources you assign to each of the clients based on your device, which are specified with `options.backend.client-resources.num-cpus` and `options.backend.client-resources.num-gpus` under `[tool.flwr.federations.local-simulation]` entry in `pyproject.toml`.


## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).
