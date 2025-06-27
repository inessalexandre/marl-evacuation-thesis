# Multi-Agent Reinforcement Learning Environment Setup

This repository contains the experimental setup for the master's thesis **"Multi-Agent Reinforcement Learning Approaches to Leverage Distributed Problem Solving"**, which investigates the use of Multi-Agent Reinforcement Learning (MARL) algorithms to efficiently and collaboratively solve distributed problems.

## ⚙️ Requirements

Make sure you have **Python 3.11** installed before starting.

Main libraries used:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [PettingZoo](https://www.pettingzoo.ml/)
- [Gymnasium](https://gymnasium.farama.org/)
- [SuperSuit](https://github.com/Farama-Foundation/SuperSuit)
- NumPy, PyTorch, Matplotlib, and other dependencies

## 🔧 Environment Setup

```bash
# Clone the repository
git clone https://github.com/inessalexandre/your-repository-name.git
cd PettingZoo

# Create and activate a virtual environment
python3.11 -m venv venv-marl
source venv-marl/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

## 🚀 Running an Experiment

After setting up the environment and activating the virtual environment:

```bash
cd pettingzoo/mpe/simple_evacuation
python train_evac.py --train --scenario 2 --algo ppo

```