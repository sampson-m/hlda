#!/bin/bash

set -e

CONFIG_FILE="dataset_identities.yaml"
HELDOUT_CELLS=600  # 20% of 3000 per identity (adjust as needed)

# List of simulation configs: DATA_ROOT, EST_ROOT, TOPIC_CONFIG, N_IDENTITY_TOPICS, DATASET_NAME
SIM_CONFIGS=(
  # Format: DATA_ROOT EST_ROOT TOPIC_CONFIG N_IDENTITY_TOPICS DATASET_NAME N_EXTRA_TOPICS N_TOPICS
  "data/ABCD_V1V2_new estimates/ABCD_V1V2_new 6_topic_fit 4 ABCD_V1V2 2 6"
)

for CONFIG in "${SIM_CONFIGS[@]}"; do
  set -- $CONFIG
  DATA_ROOT="$1"
  EST_ROOT="$2"
  TOPIC_CONFIG="$3"
  N_IDENTITY_TOPICS="$4"
  DATASET_NAME="$5"
  N_EXTRA_TOPICS="$6"
  N_TOPICS="$7"

  # Process DE_mean 0.1 to 0.5
  for DE_MEAN in "DE_mean_0.1" "DE_mean_0.2" "DE_mean_0.3" "DE_mean_0.4" "DE_mean_0.5"; do
      SIMDIR="${DATA_ROOT}/${DE_MEAN}"
      if [ ! -d "$SIMDIR" ]; then
          echo "Warning: Directory $SIMDIR does not exist, skipping..."
          continue
      fi
      
      echo "=== Processing $DE_MEAN ($DATASET_NAME) ==="
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

      # 2. Fit HLDA
      python3 scripts/shared/fit_hlda.py \
          --counts_csv "$TRAIN_CSV" \
          --n_extra_topics $N_EXTRA_TOPICS \
          --output_dir "$OUTDIR/HLDA" \
          --dataset $DATASET_NAME \
          --config_file "$CONFIG_FILE" \
          --n_loops 10000 \
          --burn_in 4000 \
          --thin 30

      # 3. Fit LDA and NMF
      python3 scripts/shared/fit_lda_nmf.py \
          --counts_csv "$TRAIN_CSV" \
          --n_topics $N_TOPICS \
          --output_dir "$OUTDIR" \
          --model both \
          --max_iter 1000

      # 4. Evaluate models
      python3 scripts/shared/evaluate_models.py \
          --counts_csv "$TRAIN_CSV" \
          --test_csv "$TEST_CSV" \
          --output_dir "$OUTDIR" \
          --n_extra_topics $N_EXTRA_TOPICS \
          --dataset $DATASET_NAME \
          --config_file "$CONFIG_FILE"

      # 5. Analyze all fits
      python3 scripts/shared/analyze_all_fits.py \
          --base_dir "$EST_ROOT/$DE_MEAN" \
          --topic_configs "$N_TOPICS" \
          --config_file "$CONFIG_FILE"

      # Restore true cell names as index in estimated theta files (for simulation only)
      for MODEL in HLDA LDA NMF; do
          EST_THETA="$OUTDIR/$MODEL/${MODEL}_theta.csv"
          TRAIN_CELLS="$SIMDIR/train_cells.csv"
          if [ -f "$EST_THETA" ] && [ -f "$TRAIN_CELLS" ]; then
              python3 <<EOF
import pandas as pd
train_theta = pd.read_csv('$EST_THETA', index_col=0)
train_cells = pd.read_csv('$TRAIN_CELLS')['cell']
if len(train_theta) == len(train_cells):
    train_theta.index = train_cells
    train_theta.to_csv('$EST_THETA')
    print('✓ Restored original cell names to train theta: $EST_THETA')
else:
    print('⚠ train_cells.csv length does not match train theta shape for $MODEL')
EOF
          fi
          TEST_THETA="$OUTDIR/model_comparison/${MODEL}_test_theta_nnls.csv"
          TEST_CELLS="$SIMDIR/test_cells.csv"
          if [ -f "$TEST_THETA" ] && [ -f "$TEST_CELLS" ]; then
              python3 <<EOF
import pandas as pd
test_theta = pd.read_csv('$TEST_THETA', index_col=0)
test_cells = pd.read_csv('$TEST_CELLS')['cell']
if len(test_theta) == len(test_cells):
    test_theta.index = test_cells
    test_theta.to_csv('$TEST_THETA')
    print('✓ Restored original cell names to test theta: $TEST_THETA')
else:
    print('⚠ test_cells.csv length does not match test theta shape for $MODEL')
EOF
          fi
      done

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
              TEST_THETA="$OUTDIR/model_comparison/${MODEL}_test_theta_nnls.csv"
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
          --n_identity_topics $N_IDENTITY_TOPICS; then
          echo "  ✓ Combined evaluation completed successfully for $DE_MEAN"
          echo "  ✓ Outputs saved to: $OUTDIR/model_comparison/"
      else
          echo "  ✗ Combined evaluation failed for $DE_MEAN"
      fi

      # 8. Run noise analysis (orthogonalization analysis)
      echo "Running noise analysis for $DE_MEAN..."
      if python3 scripts/simulation/run_noise_analysis.py; then
          echo "  ✓ Noise analysis completed successfully"
          echo "  ✓ Results saved to: $EST_ROOT/$DE_MEAN/noise_analysis/"
      else
          echo "  ✗ Noise analysis failed for $DE_MEAN"
      fi

  done
done

echo "\nAll simulation pipelines complete for ABCD_V1V2_new (DE_mean 0.1 to 0.5)!" 