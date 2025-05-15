# ising-monte-carlo-pca
Monte Carlo Sampling for 2D Ising Model with Metropolis and Wolff Algorithms and PCA analysis to discover phase transitions

Developed a Python-based Monte Carlo simulator for the 2D Ising model, implementing both Metropolis updates (suitable for high temperatures) and Wolff cluster updates (optimized for low temperatures). The simulator supports configurable lattice sizes, flexible temperature schedules, and efficient parallel generation of spin configurations using Python multiprocessing.

Generated datasets are stored in .npy format for downstream analysis (e.g., phase classification, dimensionality reduction via PCA, critical behavior analysis). The project emphasizes clean architecture, reproducibility, and scalability, serving as a strong foundation for further research in statistical mechanics and AI-driven physics modeling.

# 2D Ising Model Monte Carlo Simulation

This repository implements a Monte Carlo simulator for the 2D Ising model, using both the Metropolis algorithm and the Wolff cluster algorithm. It allows efficient generation of Ising samples for machine learning applications (e.g., PCA, supervised learning, unsupervised clustering).

## Features
- Supports both **square** and **triangular** lattices
- **Metropolis** updates for high-temperature regime
- **Wolff cluster** updates for low-temperature regime
- **Parallel sample generation** using multiprocessing
- **Easy saving** of generated samples as `.npy` files
- Simple **visualization** of lattice configurations

## Installation

```bash
pip install -r requirements.txt