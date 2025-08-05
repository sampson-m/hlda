# Improved Model Recovery Signal-to-Noise Analysis

## Overview

This document describes the improvements made to the model recovery signal-to-noise analysis, addressing the need for better visualizations and more meaningful metrics.

## Key Improvements

### 1. Median Percent Residuals Instead of Correlation

**Before**: Used correlation coefficients between true and estimated parameters
**After**: Use median percent residuals for more informative error measurement

**Benefits**:
- More intuitive interpretation (lower values = better recovery)
- Percent-based metric is more informative than absolute differences
- Median is robust to outliers compared to mean
- Avoids issues with correlation being misleading when distributions differ

**Implementation**:
- Normalize both true and estimated parameters to probabilities
- Compute percent differences: `|true_normalized - estimated_normalized| / true_normalized * 100`
- Use median across all cells/genes for each topic (robust to outliers)

### 2. Topic-Specific Recovery Plots

**Before**: Single busy plot with all topics and model types
**After**: Separate focused plots for each topic

**New Visualization Structure**:
- **Individual Topic Plots**: One plot per topic (A, B, C, D, V1, V2, etc.)
  - X-axis: DE means (0.1, 0.2, 0.3, etc.)
  - Y-axis: Average residuals
  - Different lines/markers for each model type (HLDA, LDA, NMF)
  - Clean, focused visualization for each topic

- **Combined Overview Plots**: 
  - All topics together for easy comparison
  - Separate plots for theta and beta residuals

### 3. Enhanced File Organization

**New Files Created**:
- `topic_specific_theta_recovery.png` - Individual topic plots for theta recovery
- `topic_specific_beta_recovery.png` - Individual topic plots for beta recovery
- `topic_specific_recovery_combined.png` - Combined view of all topics
- `model_recovery_analysis_residuals.png` - Updated scatter plots using residuals
- `model_recovery_with_residuals.csv` - Data with new residual metrics

## Usage

### Running the Analysis

```bash
python3 scripts/simulation/run_noise_analysis.py
```

### Output Structure

For each dataset (AB_V1, ABCD_V1V2, ABCD_V1V2_new):

```
estimates/
├── [dataset_name]/
│   ├── topic_specific_theta_recovery.png      # Individual topic plots
│   ├── topic_specific_beta_recovery.png       # Individual topic plots  
│   ├── topic_specific_recovery_combined.png   # Combined overview
│   ├── model_recovery_analysis_residuals.png  # Updated scatter plots
│   └── model_recovery_with_residuals.csv      # Data with residuals
```

### Interpreting the Results

**Topic-Specific Plots**:
- Each subplot shows one topic (A, B, C, D, V1, V2)
- X-axis shows DE means (signal strength)
- Y-axis shows average residuals (lower = better recovery)
- Different colored lines for each model type
- Clear trends show how recovery varies with signal strength

**Residual Values**:
- Range: 0% to 100% (percent error)
- Lower values indicate better parameter recovery
- Values around 1-5% indicate excellent recovery
- Values > 20% may indicate poor recovery
- Values > 50% indicate very poor recovery

## Example Interpretation

**Good Recovery Pattern**:
- Residuals decrease as DE mean increases (more signal = better recovery)
- HLDA typically shows lowest residuals (best recovery)
- LDA and NMF show similar or slightly higher residuals

**Poor Recovery Pattern**:
- Residuals remain high even at high DE means
- No clear improvement with increasing signal
- All models show similar poor performance

## Technical Details

### Residual Calculation

```python
# For theta (cell-topic proportions)
true_theta_norm = true_theta / np.sum(true_theta, axis=1, keepdims=True)
est_theta_norm = est_theta / np.sum(est_theta, axis=1, keepdims=True)
epsilon = 1e-10  # Avoid division by zero
theta_percent_residuals = np.abs(true_theta_norm - est_theta_norm) / (true_theta_norm + epsilon) * 100
median_theta_residual = np.median(theta_percent_residuals)

# For beta (gene-topic proportions)  
true_beta_norm = true_beta / np.sum(true_beta, axis=0, keepdims=True)
est_beta_norm = est_beta / np.sum(est_beta, axis=0, keepdims=True)
beta_percent_residuals = np.abs(true_beta_norm - est_beta_norm) / (true_beta_norm + epsilon) * 100
median_beta_residual = np.median(beta_percent_residuals)
```

### Topic Matching

- Uses Hungarian algorithm for optimal topic assignment
- Based on correlation between true and estimated topic profiles
- Ensures fair comparison between models

## Benefits of New Approach

1. **Cleaner Visualizations**: Each topic gets its own focused plot
2. **Better Metrics**: Residuals provide direct error measurement
3. **Easier Interpretation**: Clear trends and model comparisons
4. **Comprehensive Analysis**: Both individual and combined views
5. **Reproducible Results**: Consistent methodology across datasets

## Future Enhancements

Potential improvements for future iterations:
- Confidence intervals for residual estimates
- Statistical significance testing between models
- Interactive visualizations
- Automated trend detection and reporting 