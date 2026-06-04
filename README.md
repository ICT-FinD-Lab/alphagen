# AlphaGen

<p align="center">
    <img src="images/logo.jpg" width=275 />
</p>

Automatic formulaic alpha generation with reinforcement learning.

This repository contains the code for our paper *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning* accepted by [KDD 2023](https://kdd.org/kdd2023/), Applied Data Science (ADS) track, publically available on [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599831). Some extensions upon this work are also included in this repo.

## Repository Structure

- `/alphagen` contains the basic data structures and the essential modules for starting an alpha mining pipeline;
- `/alphagen_qlib` contains the qlib-specific APIs for data preparation;
- `/alphagen_generic` contains data structures and utils designed for our baselines, which basically follow [gplearn](https://github.com/trevorstephens/gplearn) APIs, but with modifications for quant pipeline;
- `/alphagen_llm` contains LLM client abstractions and a set of prompts useful for LLM-based alpha generation, and also provides some LLM-based automatic iterative alpha-generation routines.
- `/gplearn` and `/dso` contains modified versions of our baselines;
- `/scripts` contains several scripts for running the experiments.

## Result Reproduction

Note that you can either use our builtin alpha calculation pipeline (see Choice 1), or implement an adapter to your own pipeline (see Choice 2).

### Choice 1: Stock data preparation

Builtin pipeline requires Qlib library and local-storaged stock data.

- READ THIS! We need some of the metadata (but not the actual stock price/volume data) given by Qlib, so follow the data preparing process in [Qlib](https://github.com/microsoft/qlib#data-preparation) first.
- The actual stock data we use are retrieved from [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5), due to concerns on the timeliness and truthfulness of the data source used by Qlib.
- The data can be downloaded by running the script `data_collection/fetch_baostock_data.py`. The newly downloaded data is saved into `~/.qlib/qlib_data/cn_data_baostock_fwdadj` by default. This path can be customized to fit your specific needs, but make sure to use the correct path when loading the data (In `alphagen_qlib/stock_data.py`, function `StockData._init_qlib`, the path should be passed to qlib with `qlib.init(provider_uri=path)`).

### Choice 2: Adapt to external pipelines

Maybe you have better implements of alpha calculation, you can implement an adapter of `alphagen.data.calculator.AlphaCalculator`. The interface is defined as follows:

```python
class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        'Calculate both IC and Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'
```

Reminder: the values evaluated from different alphas may have drastically different scales, we recommend that you should normalize them before combination.

### Before running

All principle components of our expriment are located in [train_maskable_ppo.py](train_maskable_ppo.py).

These parameters may help you build an `AlphaCalculator`:

- instruments (Set of instruments)
- start_time & end_time (Data range for each dataset)
- target (Target stock trend, e.g., 20d return rate)

These parameters will define a RL run:

- batch_size (PPO batch size)
- features_extractor_kwargs (Arguments for LSTM shared net)
- device (PyTorch device)
- save_path (Path for checkpoints)
- tensorboard_log (Path for TensorBoard)

### Run the experiments

Please run the individual scripts at the root directory of this project as modules, i.e. `python -m scripts.NAME ARGS...`.
Use `python -m scripts.NAME -h` for information on the arguments.

- `scripts/rl.py`: Main experiments of AlphaGen/HARLA
- `scripts/llm_only.py`: Alpha generator based solely on iterative interactions with an LLM.
- `scripts/llm_test_validity.py`: Tests on how the system prompt affects the valid alpha rate of an LLM.

### After running

- Model checkpoints and alpha pools are located in `save_path`;
    - The model is compatiable with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
    - Alpha pools are formatted in human-readable JSON.
- Tensorboard logs are located in `tensorboard_log`.

## Baselines

### GP-based methods

[gplearn](https://github.com/trevorstephens/gplearn) implements Genetic Programming, a commonly used method for symbolic regression. We maintained a modified version of gplearn to make it compatiable with our task. The corresponding experiment scipt is [gp.py](gp.py)

### Deep Symbolic Regression

[DSO](https://github.com/brendenpetersen/deep-symbolic-optimization) is a mature deep learning framework for symbolic optimization tasks. We maintained a minimal version of DSO to make it compatiable with our task. The corresponding experiment scipt is [dso.py](dso.py)

## Trading (Experimental)

We implemented some trading strategies based on Qlib. See [backtest.py](backtest.py) and [trade_decision.py](trade_decision.py) for demos.

## Citing our work

```bibtex
@inproceedings{alphagen,
    author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and Pan, Feiyang and He, Jia and Tu, Dandan and He, Qing},
    title = {Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning},
    year = {2023},
    doi = {10.1145/3580305.3599831},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```

```bibtex
@article{HARLA,
  author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and He, Qing},
  title = {A hybrid approach to formulaic alpha discovery with large language model assistance},
  journal = {Frontiers of Computer Science},
  volume = {20},
  number = {2},
  pages = {2002316},
  year = {2026},
  doi = {10.1007/S11704-025-41061-5},
}
```

## Contributing

Feel free to submit Issues or Pull requests.

## Contributors

This work is maintained by the MLDM research group, [IIP, ICT, CAS](http://iip.ict.ac.cn/).

Maintainers include:

- [Hongyan Xue](https://github.com/xuehongyanL)
- [Shuo Yu](https://github.com/Chlorie)

Thanks to the following contributors:

- [@yigaza](https://github.com/yigaza)

Thanks to the following in-depth research on our project:

- *因子选股系列之九十五: DFQ强化学习因子组合挖掘系统*
