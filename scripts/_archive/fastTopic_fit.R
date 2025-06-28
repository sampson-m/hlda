suppressMessages(library(fastTopics))
suppressMessages(library(data.table))
suppressMessages(library(Matrix))  # for sparse matrix conversion

set.seed(123)
counts_file = "../data/hsim/simulated_counts.csv"
counts_df = fread(counts_file, data.table = FALSE)
groups = counts_df[, 1]
counts_df = counts_df[, -1, drop = FALSE]
counts = as.matrix(counts_df)
if (!is.numeric(counts)) { stop("The counts matrix is not numeric.") }
if (any(counts < 0)) { stop("The counts matrix contains negative values.") }
cat("Counts matrix dimensions:", dim(counts), "\n")

# Convert the counts matrix to a sparse matrix.
sparse_counts = Matrix(counts, sparse = TRUE)
cat("Sparse counts matrix dimensions:", dim(sparse_counts), "\n")

K = 5
cat("Fitting the topic model with", K, "topics...\n")
# Use the sparse matrix in the topic model fitting.
fit = fit_topic_model(sparse_counts, k = K)
theta = fit$L
rownames(theta) = groups
write.csv(theta, file = "../estimates/hsim/NMF_theta.csv", row.names = TRUE)
F_loadings = fit$F
write.csv(F_loadings, file = "../estimates/hsim/NMF_beta.csv", row.names = TRUE)