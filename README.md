# Reinforcement learning driven TabuCol (RLTCol)

The RLTCol algorithm works by iteratively running the local search algorithm TabuCol, and running an RL agent. The two components pass solutions to each other.
Also the project implemments Ant colony algorithm (ACO) for training and testing, which aims to provide a good starting point for tabucol algorithm

## Code

The RL agent is implemented in Python using the [Tianshou](https://github.com/thu-ml/tianshou) library. TabuCol is implemented in Rust, using [maturin](https://github.com/PyO3/maturin) to interface with Python. The code is written for Python 3.10.

### Requirements and Installation

Create a virtual environment and install the required Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, build the Rust code:

```bash
cd src/tabucol && maturin develop --release && cd -
```

You can choose RLTabucol0(without RL) or RL-Tabucol or ACO-RL-Tabucol for training or testing (The training and testing command as follows:
# run example
1: RLTabucol0: ```bash python batch_runner.py dummy.pt instance.col color_num output_dir 
--time-limit 18000 --RL False --num-jobs 10 --concurrency 2 ``` 

2: RL-Tabucol: ```bash python batch_runner.py dummy.pt instance.col color_num output_dir 
--time-limit 18000 --RL True --num-jobs 10 --concurrency 2 ``` 

3: ACO-RL-Tabucol: ```bash python batch_runner.py dummy.pt instance.col color_num output_dir 
--time-limit 18000 --RL True --num-jobs 10 --concurrency 2 ```

You can train as follows:

# train example
```bash python trainer.py -E 50 -N 250 -C 24 ```

## Usage

The graphs used as input need to be in the form of a DIMACS text file. The graphs used in the paper can be found [here](https://mat.tepper.cmu.edu/COLOR/instances.html). If they are in the compressed format, they can be decompressed using the translator found on the same page.

The source code for the RLTCol algorithm is located in the `src` directory. The TabuCol implementation in Rust is located in the `src/tabucol` directory.
