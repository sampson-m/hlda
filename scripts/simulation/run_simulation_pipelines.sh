#!/bin/bash

set -e

DATA_ROOT="data/AB_V1"
EST_ROOT="estimates/AB_V1"
CONFIG_FILE="dataset_identities.yaml"
HELDOUT_CELLS=600  # 20% of 3000 per identity (adjust as needed)
TOPIC_CONFIG="3_topic_fit"  # 2 identities + 1 activity (adjust as needed)

# Only run on DE_mean_0.1 for quick testing
for SIMDIR in ${DATA_ROOT}/DE_mean_0.1; do
    DE_MEAN=$(basename "$SIMDIR")
    echo "\n=== Processing $DE_MEAN ==="
    COUNTS_CSV="$SIMDIR/counts.csv"
    TRUE_THETA="$SIMDIR/theta.csv"
    TRUE_BETA="$SIMDIR/gene_means.csv"
    OUTDIR="$EST_ROOT/$DE_MEAN/$TOPIC_CONFIG"
    mkdir -p "$OUTDIR"

    TRAIN_CSV="$SIMDIR/filtered_counts_train.csv"
    TEST_CSV="$SIMDIR/filtered_counts_test.csv"

    # Remove any accidental header rows (index == 'cell') from train and test CSVs
    python3 -c "import pandas as pd; f='$TRAIN_CSV'; df=pd.read_csv(f, index_col=0); df = df[df.index != 'cell']; df.index = pd.Series(df.index).str.split('_').str[0]; df.to_csv(f)"
    python3 -c "import pandas as pd; f='$TEST_CSV'; df=pd.read_csv(f, index_col=0); df = df[df.index != 'cell']; df.index = pd.Series(df.index).str.split('_').str[0]; df.to_csv(f)"

    # 2. Fit HLDA (set iterations very low for testing)
    python3 scripts/shared/fit_hlda.py \
        --counts_csv "$TRAIN_CSV" \
        --n_extra_topics 1 \
        --output_dir "$OUTDIR/HLDA" \
        --dataset AB_V1 \
        --config_file "$CONFIG_FILE" \
        --n_loops 21 \
        --burn_in 5 \
        --thin 2

    # 3. Fit LDA and NMF (set iterations very low for testing)
    python3 scripts/shared/fit_lda_nmf.py \
        --counts_csv "$TRAIN_CSV" \
        --n_topics 3 \
        --output_dir "$OUTDIR" \
        --model both \
        --max_iter 10

    # Restore true cell names as index in estimated theta files (for simulation only)
    for MODEL in HLDA LDA NMF; do
        EST_THETA="$OUTDIR/$MODEL/${MODEL}_theta.csv"
        if [ -f "$EST_THETA" ]; then
            python3 <<EOF
import pandas as pd
theta = pd.read_csv('$EST_THETA', index_col=0)
cells = pd.read_csv('$SIMDIR/train_cells.csv')['cell']
if len(theta) == len(cells):
    theta.index = cells
    theta.to_csv('$EST_THETA')
else:
    print('Warning: cell mapping and theta shape mismatch for $MODEL')
EOF
        fi
    done

    # 4. Evaluate models
    python3 scripts/shared/evaluate_models.py \
        --counts_csv "$TRAIN_CSV" \
        --test_csv "$TEST_CSV" \
        --output_dir "$OUTDIR" \
        --n_extra_topics 1 \
        --dataset AB_V1 \
        --config_file "$CONFIG_FILE"

    # 5. Analyze all fits
    python3 scripts/shared/analyze_all_fits.py \
        --base_dir "$EST_ROOT/$DE_MEAN" \
        --topic_configs "3" \
        --config_file "$CONFIG_FILE"

    # 6. Theta and Beta comparison plots, plus noise analysis
    for MODEL in HLDA LDA NMF; do
        EST_THETA="$OUTDIR/$MODEL/${MODEL}_theta.csv"
        EST_BETA="$OUTDIR/$MODEL/${MODEL}_beta.csv"
        PLOT_DIR="$OUTDIR/$MODEL/plots"
        mkdir -p "$PLOT_DIR"
        
        # Theta scatter plots - train and test
        if [ -f "$EST_THETA" ]; then
            # Train data theta scatter plot
            python3 -c "
import sys
sys.path.append('scripts/simulation')
from simulation_evaluation_functions import compare_theta_true_vs_estimated
compare_theta_true_vs_estimated('$TRUE_THETA', '$EST_THETA', '$PLOT_DIR/theta_scatter_train.png', '$MODEL Train')
"
            
            # Test data theta scatter plot (using pre-computed test theta)
            TEST_THETA="$PLOT_DIR/${MODEL}_test_theta_nnls.csv"
            if [ -f "$TEST_THETA" ]; then
                python3 -c "
import sys
sys.path.append('scripts/simulation')
from simulation_evaluation_functions import compare_theta_true_vs_estimated
compare_theta_true_vs_estimated('$TRUE_THETA', '$TEST_THETA', '$PLOT_DIR/theta_scatter_test.png', '$MODEL Test')
"
            else
                echo "Warning: $TEST_THETA not found for $MODEL in $DE_MEAN"
            fi
        else
            echo "Warning: $EST_THETA not found for $MODEL in $DE_MEAN"
        fi
        
        # Beta scatter plots
        if [ -f "$EST_BETA" ]; then
            python3 -c "
import sys
sys.path.append('scripts/simulation')
from simulation_evaluation_functions import compare_beta_true_vs_estimated
compare_beta_true_vs_estimated('$TRUE_BETA', '$EST_BETA', '$PLOT_DIR/beta_scatter.png', '$MODEL')
"
        else
            echo "Warning: $EST_BETA not found for $MODEL in $DE_MEAN"
        fi
        
    done

    # 7. Generate combined scatter plots and noise analysis
    echo "Generating combined evaluation (scatter plots + noise analysis) for $DE_MEAN..."
    if python3 scripts/simulation/combined_evaluation.py \
        --simulation_dir "$SIMDIR" \
        --output_dir "$OUTDIR" \
        --models HLDA LDA NMF \
        --topic_config "$TOPIC_CONFIG" \
        --libsize_mean 1500 \
        --n_identity_topics 2; then
        echo "  ✓ Combined evaluation completed successfully for $DE_MEAN"
        echo "  ✓ Outputs saved to: $OUTDIR/model_comparison/"
    else
        echo "  ✗ Combined evaluation failed for $DE_MEAN"
    fi

done

echo "\nAll simulation pipelines complete!" 