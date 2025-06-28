import pandas as pd
import numpy as np
import sys
from cnmf import get_high_var_genes, load_df_from_npz, save_df_to_npz
import os

class scsim:
    def __init__(self, ngenes=10000, ncells=100, seed=757578,
                 mean_rate=0.3, mean_shape=0.6, libloc=11, libscale=0.2,
                 expoutprob=0.05, expoutloc=4, expoutscale=0.5, ngroups=1,
                 diffexpprob=0.1, diffexpdownprob=0.5,
                 diffexploc=0.1, diffexpscale=0.4, bcv_dispersion=0.1,
                 bcv_dof=60, ndoublets=0, groupprob=None,
                 # Single-program parameters for backward compatibility:
                 nproggenes=0, progdownprob=0.5, progdeloc=0.1,
                 progdescale=0.4, proggoups=None, progcellfrac=0.2,
                 minprogusage=0.2, maxprogusage=0.8,
                 programs=None):
        self.ngenes = ngenes
        self.ncells = ncells
        self.seed = seed
        self.mean_rate = mean_rate
        self.mean_shape = mean_shape
        self.libloc = libloc
        self.libscale = libscale
        self.expoutprob = expoutprob
        self.expoutloc = expoutloc
        self.expoutscale = expoutscale
        self.ngroups = ngroups
        self.diffexpprob = diffexpprob
        self.diffexpdownprob = diffexpdownprob
        self.diffexploc = diffexploc
        self.diffexpscale = diffexpscale
        self.bcv_dispersion = bcv_dispersion
        self.bcv_dof = bcv_dof
        self.ndoublets = ndoublets
        self.init_ncells = ncells + ndoublets

        # For backward compatibility: if no programs are provided, create one program.
        if programs is None:
            self.programs = [{
                "nproggenes": nproggenes,
                "progdownprob": progdownprob,
                "progdeloc": progdeloc,
                "progdescale": progdescale,
                "proggroups": proggoups,   # If None, will default to all groups later.
                "progcellfrac": progcellfrac,
                "minprogusage": minprogusage,
                "maxprogusage": maxprogusage
            }]
        else:
            self.programs = programs

        # Validate groupprob
        if groupprob is None:
            self.groupprob = [1 / float(self.ngroups)] * self.ngroups
        elif (len(groupprob) == self.ngroups) and (np.isclose(np.sum(groupprob), 1)):
            self.groupprob = groupprob
        else:
            sys.exit('Invalid groupprob input')

    def simulate(self):
        np.random.seed(self.seed)
        print('Simulating cells')
        self.cellparams = self.get_cell_params()
        print('Simulating gene params')
        self.geneparams = self.get_gene_params()

        # Simulate programs if any program has nproggenes > 0
        if any(prog.get('nproggenes', 0) > 0 for prog in self.programs):
            print('Simulating programs')
            self.simulate_programs()

        print('Simulating DE')
        self.sim_group_DE()

        print('Simulating cell-gene means')
        self.cellgenemean = self.get_cell_gene_means()
        if self.ndoublets > 0:
            print('Simulating doublets')
            self.simulate_doublets()

        print('Adjusting means')
        self.adjust_means_bcv()
        print('Simulating counts')
        self.simulate_counts()

        print('Building theta matrix')
        self.build_theta_matrix()

    def simulate_counts(self):
        '''Sample read counts for each gene x cell from a Poisson distribution using the adjusted mean'''
        self.counts = pd.DataFrame(np.random.poisson(lam=self.updatedmean),
                                   index=self.cellnames, columns=self.genenames)

    def adjust_means_bcv(self):
        '''Adjust cellgenemean to follow a mean-variance trend relationship'''
        self.bcv = self.bcv_dispersion + (1 / np.sqrt(self.cellgenemean))
        chisamp = np.random.chisquare(self.bcv_dof, size=self.ngenes)
        self.bcv = self.bcv * np.sqrt(self.bcv_dof / chisamp)
        self.updatedmean = np.random.gamma(shape=1 / (self.bcv ** 2),
                                           scale=self.cellgenemean * (self.bcv ** 2))
        self.bcv = pd.DataFrame(self.bcv, index=self.cellnames, columns=self.genenames)
        self.updatedmean = pd.DataFrame(self.updatedmean, index=self.cellnames,
                                        columns=self.genenames)

    def simulate_doublets(self):
        '''Simulate doublets by merging gene means from two cells while preserving library size'''
        d_ind = sorted(np.random.choice(self.ncells, self.ndoublets, replace=False))
        d_ind = ['Cell%d' % (x + 1) for x in d_ind]
        self.cellparams['is_doublet'] = False
        self.cellparams.loc[d_ind, 'is_doublet'] = True
        extraind = self.cellparams.index[-self.ndoublets:]
        group2 = self.cellparams.loc[extraind, 'group'].values
        self.cellparams['group2'] = -1
        self.cellparams.loc[d_ind, 'group2'] = group2

        # Update cell-gene means for doublets while preserving library size
        dmean = self.cellgenemean.loc[d_ind, :].values
        dmultiplier = 0.5 / dmean.sum(axis=1)
        dmean = np.multiply(dmean, dmultiplier[:, np.newaxis])
        omean = self.cellgenemean.loc[extraind, :].values
        omultiplier = 0.5 / omean.sum(axis=1)
        omean = np.multiply(omean, omultiplier[:, np.newaxis])
        newmean = dmean + omean
        libsize = self.cellparams.loc[d_ind, 'libsize'].values
        newmean = np.multiply(newmean, libsize[:, np.newaxis])
        self.cellgenemean.loc[d_ind, :] = newmean
        # Remove extra doublet cells from the data structures
        self.cellgenemean.drop(extraind, axis=0, inplace=True)
        self.cellparams.drop(extraind, axis=0, inplace=True)
        self.cellnames = self.cellnames[0:self.ncells]

    def get_cell_gene_means(self):
        '''Calculate each gene's mean expression for each cell, adjusting for library size and program contributions'''
        # Baseline: group-specific gene means
        group_cols = [col for col in self.geneparams.columns if '_genemean' in col and 'group' in col]
        group_genemean = self.geneparams[group_cols].T.astype(float)
        group_genemean = group_genemean.div(group_genemean.sum(axis=1), axis=0)
        ind = self.cellparams['group'].apply(lambda x: f'group{x}_genemean')
        baseline = group_genemean.loc[ind, :].astype(float)
        baseline.index = self.cellparams.index

        # Combine program contributions if any
        total_prog_usage = pd.Series(0, index=baseline.index, dtype=float)
        prog_contrib = pd.DataFrame(0, index=baseline.index, columns=baseline.columns, dtype=float)
        for i, prog in enumerate(self.programs, start=1):
            # Get the usage for program i; if not present, default to 0.
            usage_series = self.cellparams.get(f'program_usage_{i}', pd.Series(0.0, index=baseline.index))
            total_prog_usage = total_prog_usage.add(usage_series, fill_value=0.0)
            # Use program-specific gene means if available.
            if f'prog_genemean_{i}' in self.geneparams.columns:
                prog_mean = self.geneparams[f'prog_genemean_{i}']
                prog_mean_norm = prog_mean / prog_mean.sum()
                prog_contrib_i = pd.DataFrame(np.outer(usage_series.values, prog_mean_norm.values),
                                              index=baseline.index,
                                              columns=baseline.columns)
                prog_contrib = prog_contrib.add(prog_contrib_i, fill_value=0.0)

        # Final cell gene mean: baseline scaled by (1 - total program usage) plus program contributions.
        cellgenemean = baseline.multiply(1 - total_prog_usage, axis=0) + prog_contrib

        # Normalize by cell library size.
        normfac = (self.cellparams['libsize'] / cellgenemean.sum(axis=1)).values
        cellgenemean = cellgenemean.multiply(normfac, axis=0)
        return cellgenemean

    def get_gene_params(self):
        '''Sample each gene's mean expression from a gamma distribution and flag outlier genes'''
        basegenemean = np.random.gamma(shape=self.mean_shape,
                                       scale=1.0 / self.mean_rate,
                                       size=self.ngenes)
        is_outlier = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.expoutprob, 1 - self.expoutprob])
        outlier_ratio = np.ones(self.ngenes)
        outliers = np.random.lognormal(mean=self.expoutloc,
                                       sigma=self.expoutscale,
                                       size=is_outlier.sum())
        outlier_ratio[is_outlier] = outliers
        gene_mean = basegenemean.copy()
        median = np.median(basegenemean)
        gene_mean[is_outlier] = outliers * median
        self.genenames = ['Gene%d' % i for i in range(1, self.ngenes + 1)]
        geneparams = pd.DataFrame({
            'BaseGeneMean': basegenemean,
            'is_outlier': is_outlier,
            'outlier_ratio': outlier_ratio,
            'gene_mean': gene_mean
        }, index=self.genenames)
        # Also create group-specific gene mean columns (one per group)
        for g in range(1, self.ngroups + 1):
            # Here, using the overall gene_mean as the baseline for each group.
            geneparams[f'group{g}_genemean'] = geneparams['gene_mean']
        return geneparams

    def get_cell_params(self):
        '''Sample cell group identities and library sizes'''
        groupid = self.simulate_groups()
        libsize = np.random.lognormal(mean=self.libloc, sigma=self.libscale,
                                      size=self.init_ncells)
        self.cellnames = ['Cell%d' % i for i in range(1, self.init_ncells + 1)]
        cellparams = pd.DataFrame({
            'group': groupid,
            'libsize': libsize
        }, index=self.cellnames)
        cellparams['group'] = cellparams['group'].astype(int)
        return cellparams

    def simulate_programs(self):
        '''Simulate multiple activity programs, selecting an independent set of genes for each program'''
        for i, prog in enumerate(self.programs, start=1):
            nprog = prog.get('nproggenes', 0)
            if nprog <= 0:
                continue

            # Randomly select a set of genes for this program.
            proggenes = np.random.choice(self.genenames, size=nprog, replace=False)
            # Create a boolean flag for these program genes.
            self.geneparams[f'prog_gene_{i}'] = False
            self.geneparams.loc[proggenes, f'prog_gene_{i}'] = True

            # Gene-level simulation: calculate DE ratios for selected program genes.
            DEratio = np.random.lognormal(mean=prog['progdeloc'],
                                          sigma=prog['progdescale'],
                                          size=nprog)
            # Probability of downregulation
            is_down = np.random.choice([True, False], size=nprog,
                                       p=[prog.get('progdownprob', 0.0),
                                          1 - prog.get('progdownprob', 0.0)])
            DEratio[is_down] = 1.0 / DEratio[is_down]

            # Start with a vector of ones for all genes and then assign DE ratios for selected genes.
            all_DE_ratio = np.ones(self.ngenes)
            gene_mask = self.geneparams.index.isin(proggenes)
            all_DE_ratio[gene_mask] = DEratio
            prog_mean = self.geneparams['gene_mean'] * all_DE_ratio
            self.geneparams[f'prog_genemean_{i}'] = prog_mean

            # Cell-level simulation: assign program usage to cells in specified groups.
            groups = prog.get('proggroups')
            if groups is None:
                groups = np.arange(1, self.ngroups + 1)
            self.cellparams[f'has_program_{i}'] = False
            self.cellparams[f'program_usage_{i}'] = 0.0
            for g in groups:
                group_cells = self.cellparams.index[self.cellparams['group'] == g]
                hasprog = np.random.choice([True, False], size=len(group_cells),
                                           p=[prog['progcellfrac'], 1 - prog['progcellfrac']])
                self.cellparams.loc[group_cells[hasprog], f'has_program_{i}'] = True
                usage = np.random.uniform(low=prog['minprogusage'],
                                          high=prog['maxprogusage'],
                                          size=hasprog.sum())
                self.cellparams.loc[group_cells[hasprog], f'program_usage_{i}'] = usage

        # Normalize usage so that total usage per cell does not exceed 1
        prog_usage_cols = [col for col in self.cellparams.columns if col.startswith("program_usage_")]
        total_usage = self.cellparams[prog_usage_cols].sum(axis=1)
        mask = total_usage > 1
        if mask.any():
            self.cellparams.loc[mask, prog_usage_cols] = (
                self.cellparams.loc[mask, prog_usage_cols].div(total_usage[mask], axis=0)
            )

    def simulate_groups(self):
        '''Sample cell group identities from a categorical distribution'''
        groupid = np.random.choice(np.arange(1, self.ngroups + 1),
                                   size=self.init_ncells, p=self.groupprob)
        self.groups = np.unique(groupid)
        return groupid

    def sim_group_DE(self):
        '''Sample differentially expressed genes and the DE factor for each cell-type group'''
        groups = self.cellparams['group'].unique()
        # Exclude any genes marked as program genes from DE between groups.
        prog_flags = [self.geneparams[col] for col in self.geneparams.columns if col.startswith('prog_gene_')]
        if prog_flags:
            proggene = np.any(np.vstack(prog_flags), axis=0)
        else:
            proggene = np.array([False] * self.geneparams.shape[0])
        for group in self.groups:
            isDE = np.random.choice([True, False], size=self.ngenes,
                                    p=[self.diffexpprob, 1 - self.diffexpprob])
            isDE[proggene] = False  # Exclude program genes from being differentially expressed.
            DEratio = np.random.lognormal(mean=self.diffexploc,
                                          sigma=self.diffexpscale,
                                          size=isDE.sum())
            DEratio[DEratio < 1] = 1. / DEratio[DEratio < 1]
            is_down = np.random.choice([True, False], size=len(DEratio),
                                       p=[self.diffexpdownprob, 1 - self.diffexpdownprob])
            DEratio[is_down] = 1. / DEratio[is_down]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[isDE] = DEratio
            group_mean = self.geneparams['gene_mean'] * all_DE_ratio
            deratiocol = f'group{group}_DEratio'
            groupmeancol = f'group{group}_genemean'
            self.geneparams[deratiocol] = all_DE_ratio
            self.geneparams[groupmeancol] = group_mean

    # ------------------------------------------------------------------
    # New function to build the theta matrix
    # ------------------------------------------------------------------
    def build_theta_matrix(self):
        """
        Build a theta matrix that includes columns for each group and each program.
        For each cell:
          - The column corresponding to the cell's group is set to (1 - sum_of_program_usages).
          - Each program column is set to the cell's program usage.
          - All other group columns are 0 for that cell.
        The result is stored in self.theta (and returned).
        """
        # Identify groups and program usage columns
        groups = sorted(self.cellparams["group"].unique())
        program_usage_cols = sorted(
            [c for c in self.cellparams.columns if c.startswith("program_usage_")]
        )

        # Construct the column order: all groups first, then all program usage columns
        theta_cols = [f"group_{g}" for g in groups] + program_usage_cols

        # Initialize a DataFrame of zeros
        theta = pd.DataFrame(0.0, index=self.cellparams.index, columns=theta_cols)

        for cell in self.cellparams.index:
            # Which group does this cell belong to?
            g = self.cellparams.loc[cell, "group"]
            group_col = f"group_{g}"

            # Sum of program usages for this cell
            sum_prog_usage = self.cellparams.loc[cell, program_usage_cols].sum()

            # Identity usage = 1 - sum_prog_usage (clamped at 0 if there's floating rounding)
            identity_usage = max(0.0, 1.0 - sum_prog_usage)
            theta.at[cell, group_col] = identity_usage

            # Copy program usage
            for pcol in program_usage_cols:
                theta.at[cell, pcol] = self.cellparams.at[cell, pcol]

        self.theta = theta
        return theta


def load_npz_as_dataframe(filepath):
    with np.load(filepath, allow_pickle=True) as npz_file:
        data = npz_file["data"]
        row_labels = npz_file["index"]
        col_labels = npz_file["columns"]
    return pd.DataFrame(data=data, index=row_labels, columns=col_labels)

def get_filenames(simdir):
    countfn = os.path.join(simdir, "counts.npz")
    countfilt_fn = os.path.join(simdir, "counts_filt.npz")
    TPM_fn = os.path.join(simdir, "TPM.npz")
    genestats_fn = os.path.join(simdir, "genestats.txt")
    highvargenes_fn = os.path.join(simdir, "highvargenes.txt")
    return countfn, countfilt_fn, TPM_fn, genestats_fn, highvargenes_fn

# Helper function: filter simulation counts and save outputs.
def process_simulation_output(simdir, min_counts_per_cell, min_cellfrac_per_gene):
    countfn, countfilt_fn, TPM_fn, genestats_fn, highvargenes_fn = get_filenames(simdir)

    # Skip processing if output already exists.
    if os.path.exists(highvargenes_fn):
        return

    print(f"Processing simulation output in {simdir}")
    counts = load_df_from_npz(countfn)
    
    # Define filtering thresholds.
    numNonZeroCells = counts.shape[0] * min_cellfrac_per_gene
    TPM = counts.div(counts.sum(axis=1), axis=0)
    counts_per_cell = counts.sum(axis=1)
    num_cells_per_gene = (counts > 0).sum(axis=0)
    
    genes2drop = num_cells_per_gene.index[num_cells_per_gene < numNonZeroCells]
    cells2drop = counts_per_cell.index[counts_per_cell < min_counts_per_cell]
    
    counts.drop(cells2drop, axis=0, inplace=True)
    counts.drop(genes2drop, axis=1, inplace=True)
    
    save_df_to_npz(counts, countfilt_fn)
    
    TPM.drop(cells2drop, axis=0, inplace=True)
    TPM.drop(genes2drop, axis=1, inplace=True)
    save_df_to_npz(TPM, TPM_fn)
    
    # Identify high-variable genes.
    genestats, gene_fano_parameters = get_high_var_genes(TPM, minimal_mean=0.0, numgenes=2000)
    genestats.to_csv(genestats_fn, sep='\t')
    highvar_genes = genestats.index[genestats['high_var']]
    
    with open(highvargenes_fn, 'w') as f:
        f.write('\n'.join(highvar_genes))
        
    theta_path = os.path.join(simdir, "theta.npz")
    theta_df = load_npz_as_dataframe(theta_path)
    theta_filt = theta_df.loc[counts.index]
    theta_filt.to_csv(os.path.join(simdir, "theta_filt.csv"))

def save_final_outputs(simdir, seed):
    counts_filt_path = os.path.join(simdir, "counts_filt.npz")
    highvar_genes_path = os.path.join(simdir, "highvargenes.txt")
    cellparams_path = os.path.join(simdir, "cellparams.npz")
    genepath = os.path.join(simdir, "geneparams.npz")
    
    counts_filt_df = load_npz_as_dataframe(counts_filt_path)
    cellparams_df = load_npz_as_dataframe(cellparams_path)
    # Ensure we keep only the cells present in counts.
    cellparams_df = cellparams_df.loc[counts_filt_df.index]
    gene_df = load_npz_as_dataframe(genepath)
    
    # Optionally, reassign counts index to the group labels.
    counts_filt_df.index = cellparams_df["group"]
    cellparams_csv = os.path.join(simdir, "cellparams.csv")
    geneparams_csv = os.path.join(simdir, "geneparams.csv")
    
    cellparams_df.to_csv(cellparams_csv)
    gene_df.to_csv(geneparams_csv)
    
    with open(highvar_genes_path, "r") as f:
        highvar_genes = [line.strip() for line in f if line.strip()]
    
    filtered_counts_df = counts_filt_df.loc[:, counts_filt_df.columns.intersection(highvar_genes)]
    filtered_counts_df.to_csv(os.path.join(simdir, "counts_filt_highvar.csv"))
    
    print(f"Final outputs saved for seed {seed} in {simdir}")

def run_scsim():
    # ----------------------------
    # Simulation parameters
    # ----------------------------
    ngenes = 25000
    ncells = 15000
    ndoublets = 0
    ngroups = 10  # 10 identity groups
    deprob = 0.025
    libloc = 7.64
    libscale = 0.78
    mean_rate = 7.68
    mean_shape = 0.34
    expoutprob = 0.00286
    expoutloc = 6.15
    expoutscale = 0.49
    diffexpprob = deprob
    diffexpdownprob = 0.0
    diffexploc = 1.0
    diffexpscale = 1.0
    bcv_dispersion = 0.448
    bcv_dof = 22.087

    # Define two program specifications.
    programs = [
        {
            "nproggenes": 1000,
            "progdownprob": 0.0,
            "progdeloc": 1.0,
            "progdescale": 1.0,
            "proggroups": [1, 2, 3, 4, 5],
            "progcellfrac": 0.35,
            "minprogusage": 0.1,
            "maxprogusage": 0.7
        },
        {
            "nproggenes": 1000,
            "progdownprob": 0.5,
            "progdeloc": 1.0,
            "progdescale": 1.0,
            "proggroups": [6, 7, 8, 9, 10],
            "progcellfrac": 0.35,
            "minprogusage": 0.1,
            "maxprogusage": 0.7
        }
    ]
    
    # Output directory structure and simulation seeds.
    outdirbase = '../data/scsim/deloc_%.2f/Seed_%d'
    deloc = 1.0
    simseeds = [1]

    # ----------------------------
    # Run the simulation(s)
    # ----------------------------
    for seed in simseeds:
        simdir = outdirbase % (deloc, seed)
        os.makedirs(simdir, exist_ok=True)
        
        simulator = scsim(ngenes=ngenes, ncells=ncells, ngroups=ngroups,
                          libloc=libloc, libscale=libscale,
                          mean_rate=mean_rate, mean_shape=mean_shape,
                          expoutprob=expoutprob, expoutloc=expoutloc, expoutscale=expoutscale,
                          diffexpprob=diffexpprob, diffexpdownprob=diffexpdownprob,
                          diffexploc=diffexploc, diffexpscale=diffexpscale,
                          bcv_dispersion=bcv_dispersion, bcv_dof=bcv_dof,
                          ndoublets=ndoublets, programs=programs, seed=seed)
        simulator.simulate()

        # Save raw simulation outputs.
        save_df_to_npz(simulator.cellparams, os.path.join(simdir, 'cellparams'))
        save_df_to_npz(simulator.geneparams, os.path.join(simdir, 'geneparams'))
        save_df_to_npz(simulator.counts, os.path.join(simdir, 'counts'))
        save_df_to_npz(simulator.theta, os.path.join(simdir, 'theta'))

    # ----------------------------
    # Post-simulation processing
    # ----------------------------
    min_counts_per_cell = 1000
    min_cellfrac_per_gene = 1 / 500.0

    for seed in simseeds:
        simdir = outdirbase % (deloc, seed)
        process_simulation_output(simdir, min_counts_per_cell, min_cellfrac_per_gene)

    # ----------------------------
    # Save final outputs as CSV files.
    # ----------------------------
    # Here we use the last seed's directory, or loop over seeds as needed.
    seed = simseeds[-1]
    simdir = outdirbase % (deloc, seed)
    save_final_outputs(simdir, seed)

if __name__ == "__main__":
    run_scsim()
