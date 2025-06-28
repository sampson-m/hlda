# Hierarchical Latent Dirichlet Allocation (HLDA) for Single-Cell Data

This repository contains a comprehensive topic modeling pipeline for single-cell RNA sequencing data, implementing Hierarchical Latent Dirichlet Allocation (HLDA) alongside traditional LDA and NMF approaches.

## Overview

The pipeline is designed to identify cell type-specific gene expression patterns and activity states in single-cell data. It includes:

- **HLDA**: Hierarchical Latent Dirichlet Allocation with Gibbs sampling
- **LDA**: Traditional Latent Dirichlet Allocation
- **NMF**: Non-negative Matrix Factorization
- **Evaluation**: Comprehensive model comparison and visualization

## Features

- **Hierarchical Topic Structure**: Cell type identity topics + activity topics
- **Gibbs Sampling**: Robust Bayesian inference for HLDA
- **Model Comparison**: Side-by-side evaluation of HLDA, LDA, and NMF
- **Visualization**: Structure plots, PCA projections, cosine similarity matrices
- **Top Genes Extraction**: Identify most important genes per topic
- **Convergence Diagnostics**: Geweke diagnostics for MCMC chains

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd hlda
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Your data should be in CSV format with:
- Rows: Cells (index should contain cell type information)
- Columns: Genes
- Values: Raw or normalized counts

Example:
```csv
cell_id,gene1,gene2,gene3,...
T_cells_1,10,5,20,...
CD19_B_1,2,15,8,...
...
```

### 2. Run the Complete Pipeline

```bash
# Navigate to scripts directory
cd scripts

# Run all models with example data
python run_all_models.py \
    --counts_csv "../data/pbmc_filtered_counts.csv" \
    --test_csv "../data/pbmc_test_counts.csv" \
    --output_dir "../estimates/pbmc/7_topic_fit" \
    --identity_topics "T cells,CD19+ B,CD56+ NK,CD34+,Dendritic,CD14+ Monocyte" \
    --n_extra_topics 2
```

### 3. Individual Model Runs

```bash
# HLDA only
python fit_hlda.py \
    --counts_csv "../data/pbmc_filtered_counts.csv" \
    --n_extra_topics 2 \
    --output_dir "../estimates/pbmc/7_topic_fit/HLDA"

# LDA and NMF
python fit_lda_nmf.py \
    --counts_csv "../data/pbmc_filtered_counts.csv" \
    --n_topics 8 \
    --output_dir "../estimates/pbmc/7_topic_fit" \
    --model both

# Evaluation only
python evaluate_models.py \
    --counts_csv "../data/pbmc_filtered_counts.csv" \
    --test_csv "../data/pbmc_test_counts.csv" \
    --output_dir "../estimates/pbmc/7_topic_fit" \
    --identity_topics "T cells,CD19+ B,CD56+ NK,CD34+,Dendritic,CD14+ Monocyte" \
    --n_extra_topics 2
```

## Project Structure

```
hlda/
├── scripts/                    # Main analysis scripts
│   ├── fit_hlda.py            # HLDA Gibbs sampling
│   ├── fit_lda_nmf.py         # LDA and NMF fitting
│   ├── evaluate_models.py     # Model evaluation and visualization
│   ├── run_all_models.py      # Combined pipeline runner
│   ├── preprocess_h5ad.py     # Data preprocessing utilities
│   └── example_run_7_topic_fit.sh  # Example execution script
├── data/                      # Input data (not in repo)
├── estimates/                 # Model outputs (not in repo)
│   └── pbmc/
│       └── 7_topic_fit/
│           ├── HLDA/
│           ├── LDA/
│           └── NMF/
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## Model Parameters

### HLDA
- **Iterations**: 10,000 Gibbs sweeps
- **Burn-in**: 4,000 iterations
- **Thinning**: 20 (301 saved samples)
- **Topics**: Identity topics (one per cell type) + activity topics (V1, V2, ...)

### LDA/NMF
- **Max iterations**: 500
- **Topics**: Total = identity topics + activity topics

## Output Files

Each model generates:
- `*_beta.csv`: Gene-topic distributions
- `*_theta.csv`: Cell-topic distributions
- `plots/`: Visualizations and diagnostics
  - Structure plots
  - PCA projections
  - Cosine similarity matrices
  - Geweke diagnostics (HLDA)
  - Top genes per topic
  - SSE evaluation results

## Key Features

### Hierarchical Topic Structure
HLDA implements a hierarchical structure where:
- Each cell type has its own identity topic
- All cell types can access shared activity topics
- This captures both cell type-specific and shared expression patterns

### Top Genes Extraction
Automatically extracts the top 10 genes by probability for each topic:
```csv
Gene_1,T cells,CD19+ B,V1,V2
Gene_2,CD56+ NK,CD34+,...
...
Gene_10,...,...
```

### Convergence Diagnostics
Geweke diagnostics for HLDA MCMC chains to assess convergence.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- CVXPY
- Numba

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hlda_single_cell,
  title={Hierarchical Latent Dirichlet Allocation for Single-Cell Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hlda}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]. 