# Evaluation for General NLP challenge

We build up a multi-task language understanding pipeline to evaluate our fined-tuned LLMs.
The [MMLU](https://huggingface.co/datasets/lukaemon/mmlu) dataset is used for this evaluation, encompassing three categories: STEM, social sciences (SS), and humanities.


## Environment Setup

Create a new Python environment (we recommend Python 3.10), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -r requirements.txt

# Log in HuggingFace account
huggingface-cli login
```

## Generate model decision & calculate accuracy

> [!NOTE]
> Please ensure that you use `quantization=4` to run the evaluation if you wish to participate in the LLM Leaderboard.

The model answers and accuracy values will be saved to `benchmarks/generation_{dataset_name}_{category_name}_{run_name}.jsonl` and `benchmarks/acc_{dataset_name}_{category_name}_{run_name}.txt`, respectively.

```bash
cd /your_project_path/NI-FlowerTune-LLM-Leaderboard-GeneralNLP-SmolLM2-Series
bash script.sh
```

### Benchmark

| Challenges                      | STEM       |   Social   |  Humanities |   Avg      |
| :--------:                      | :--------: | :--------: | :--------:  | :--------: |
|SmolLM2-135M-Instruct (100Rounds)| 18.68      |  21.90     |  22.74      |   21.10    |
|SmolLM2-135M (100Rounds)         | 2.94       |  3.31      |  2.29       |   2.84     |
|SmolLM2-360M-Instruct (100Rounds)| 19.44      |  19.43     |  12.68      |   17.18    |
|SmolLM2-360M (100Rounds)         |            |            |             |            |

> [!NOTE]
> Please ensure that you provide all **three accuracy values (STEM, SS, Humanities)** for three evaluation categories when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
