"""Run Replogle raw-data selection, PCA, IV/SV/GMM, and Wald/local inference.

Usage: python code/experiments/replogle/run_replogle.py
All generated data go to results/replogle. Visualizations belong in the notebook.
"""

# Original notebook computation cell 1
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA

CONTROL = 'control'
MIN_SAMPLES = 30
TOP_K = 20
N_OUTCOMES = 2000
N_NON_DE = 1000
N_SAVED_PCS = 100
N_PCS = 10
ALPHA = 0.05
# Local misspecification set: tau = diag(r_iv, r_sv) @ u, ||u||_2 <= 1.
# Each radius is kappa times the clustered SD of its moment, estimated at GMM beta.
LOCAL_KAPPA_IV = 0.5
LOCAL_KAPPA_SV = 0.5
if min(LOCAL_KAPPA_IV, LOCAL_KAPPA_SV) < 0:
    raise ValueError('Local misspecification radii must be nonnegative.')

METHODS = {'iv': 'IV', 'sv': 'SV', 'gmm': 'Two-step GMM'}
PC_COLUMNS = [f'PC{i}' for i in range(1, N_PCS + 1)]
RAW_PVALUE_LAYER_CANDIDATES = ('pvalue', 'pvalues', 'p_value', 'p_values', 'pvalue_raw', 'pvalues_raw', 'raw_pvalue', 'raw_pvalues')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / 'data'
PSEUDOBULK_PATH = INPUT_DIR / 'Replogle-E-k562_mean_log1p_by_batch.h5ad'
DE_PATH = INPUT_DIR / 'Replogle-E-k562_wilcoxon_batch_stratified.h5ad'
for input_path in (PSEUDOBULK_PATH, DE_PATH):
    if not input_path.is_file():
        raise FileNotFoundError(
            f'Missing input: {input_path}. Prepare the precomputed H5AD inputs '
            'described in data/README.md, then rerun this script.'
        )
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'replogle'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP20_PATH = OUTPUT_DIR / 'top20_de_genes_by_perturbation.csv.gz'
NON_DE_PATH = OUTPUT_DIR / 'common_1000_non_de_genes.csv'
GLOBAL_TOP20_PATH = OUTPUT_DIR / 'global_top20_de_genes.csv'
OUTCOME_GENES_PATH = OUTPUT_DIR / 'outcome_genes.csv'
PCA_SCORES_PATH = OUTPUT_DIR / 'pca_scores.csv.gz'
PCA_LOADINGS_PATH = OUTPUT_DIR / 'pca_loadings.csv.gz'
PCA_VARIANCE_PATH = OUTPUT_DIR / 'pca_explained_variance.csv'
RESULTS_PATH = OUTPUT_DIR / 'iv_sv_gmm_results.csv.gz'
FIRST_STAGE_PATH = OUTPUT_DIR / 'first_stage_results.csv'
SKIPPED_PATH = OUTPUT_DIR / 'skipped_models.csv'

def as_dense(matrix, dtype=float):
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)

pseudobulk = ad.read_h5ad(PSEUDOBULK_PATH, backed='r')
de = ad.read_h5ad(DE_PATH, backed='r')

if not pseudobulk.var_names.equals(de.var_names):
    raise ValueError('Pseudobulk and Wilcoxon data do not contain the same ordered genes.')

required_layers = {'pvalue_adj', 'z_score', 'logfoldchanges'}
missing_layers = required_layers - set(de.layers.keys())
if missing_layers:
    raise KeyError(f'Missing required DE layers: {sorted(missing_layers)}')

layer_lookup = {str(name).lower(): str(name) for name in de.layers.keys()}
WILCOXON_RAW_P_LAYER = next((layer_lookup[name] for name in RAW_PVALUE_LAYER_CANDIDATES if name in layer_lookup), None)
if WILCOXON_RAW_P_LAYER is None:
    raise KeyError(f'No raw Wilcoxon p-value layer was found. Expected one of {RAW_PVALUE_LAYER_CANDIDATES}; available layers: {sorted(map(str, de.layers.keys()))}')

sample_counts = pseudobulk.obs['perturbation'].astype(str).value_counts().rename_axis('perturbation').reset_index(name='n_samples')

print('Pseudobulk expression:', pseudobulk.shape)
print('Wilcoxon DE:', de.shape)
print('Batches:', pseudobulk.obs['batch'].nunique())
print('Raw Wilcoxon p-value layer:', WILCOXON_RAW_P_LAYER)
print('Output folder:', OUTPUT_DIR)

# Original notebook computation cell 3
genes = de.var_names.astype(str).to_numpy()
perturbations = de.obs['perturbation'].astype(str).to_numpy()

if pd.Index(genes).has_duplicates:
    raise ValueError('Duplicated genes were found in the expression data.')
if pd.Index(perturbations).has_duplicates:
    raise ValueError('Duplicated perturbations were found in the Wilcoxon data.')

expression_gene_lookup = pd.Series(np.arange(pseudobulk.n_vars), index=pseudobulk.var_names.astype(str))
expression_genes = set(expression_gene_lookup.index)
de_perturbation_set = set(perturbations)

eligible_labels = set(sample_counts.loc[sample_counts['n_samples'].ge(MIN_SAMPLES), 'perturbation'].astype(str))
missing_de_results = sorted((eligible_labels - {CONTROL}) - de_perturbation_set)
missing_perturbation_genes = sorted((eligible_labels - {CONTROL}) - expression_genes)
eligible_labels -= set(missing_de_results) | set(missing_perturbation_genes)

if CONTROL not in eligible_labels:
    raise ValueError(f'{CONTROL!r} does not have at least {MIN_SAMPLES} pseudobulk observations.')

eligible_indices = np.flatnonzero(np.isin(perturbations, list(eligible_labels - {CONTROL})))
analysis_perturbations = perturbations[eligible_indices].astype(str).tolist()
pca_labels = {CONTROL, *analysis_perturbations}
n_perturbations, n_genes = len(analysis_perturbations), len(genes)

if not n_perturbations:
    raise ValueError('No eligible perturbations remained after filtering.')

rank_sum = np.zeros(n_genes, dtype=np.float64)
mean_abs_z = np.zeros(n_genes, dtype=np.float64)
mean_abs_lfc = np.zeros(n_genes, dtype=np.float64)
top20_frequency = np.zeros(n_genes, dtype=np.int64)
rank_values = np.arange(1, n_genes + 1, dtype=np.float64)
top20_records = []

for perturbation_index in eligible_indices:
    perturbation = perturbations[perturbation_index]
    p_adj = as_dense(de.layers['pvalue_adj'][perturbation_index], dtype=np.float64).reshape(-1)
    z_score = as_dense(de.layers['z_score'][perturbation_index], dtype=np.float64).reshape(-1)
    lfc = as_dense(de.layers['logfoldchanges'][perturbation_index], dtype=np.float64).reshape(-1)

    p_sort = np.nan_to_num(p_adj, nan=1.0, posinf=1.0, neginf=0.0)
    abs_z = np.nan_to_num(np.abs(z_score), nan=0.0)
    abs_lfc = np.nan_to_num(np.abs(lfc), nan=0.0)
    order = np.lexsort((-abs_lfc, -abs_z, p_sort))

    rank_sum[order] += rank_values
    mean_abs_z += abs_z
    mean_abs_lfc += abs_lfc

    top_indices = order[:TOP_K]
    top20_frequency[top_indices] += 1
    top20_records.extend({'perturbation': perturbation, 'rank': rank, 'gene': genes[index], 'pvalue_adj': p_adj[index], 'z_score': z_score[index], 'logfoldchange': lfc[index]} for rank, index in enumerate(top_indices, start=1))

average_de_rank = rank_sum / n_perturbations
mean_abs_z /= n_perturbations
mean_abs_lfc /= n_perturbations

top20 = pd.DataFrame(top20_records)
if not top20.groupby('perturbation', observed=True).size().eq(TOP_K).all():
    raise ValueError(f'Each perturbation must contribute exactly {TOP_K} Top-DE genes.')
top20.to_csv(TOP20_PATH, index=False, compression='gzip')

print(f'Eligible model perturbations: {n_perturbations:,}; PCA labels: {n_perturbations:,} perturbations plus control.')
print(f'Excluded for missing Wilcoxon results: {len(missing_de_results):,}; excluded for missing target expression: {len(missing_perturbation_genes):,}.')
print(f'Saved {len(top20):,} Top-DE rows to {TOP20_PATH.name}.')

# Original notebook computation cell 5
summary = pd.DataFrame({'gene': genes, 'average_de_rank': average_de_rank, 'top20_frequency': top20_frequency, 'mean_abs_z_score': mean_abs_z, 'mean_abs_logfoldchange': mean_abs_lfc})
perturbation_target_set = set(analysis_perturbations)
eligible_non_de = (summary['top20_frequency'].eq(0) & ~summary['gene'].isin(perturbation_target_set)).to_numpy()
selection_order = np.lexsort((summary['gene'].to_numpy(), summary['mean_abs_logfoldchange'].to_numpy(), summary['mean_abs_z_score'].to_numpy(), -summary['average_de_rank'].to_numpy()))
selected_indices = selection_order[eligible_non_de[selection_order]][:N_NON_DE]

if selected_indices.size < N_NON_DE:
    raise ValueError(f'Only {selected_indices.size:,} eligible non-DE genes remain; {N_NON_DE:,} are required.')

common_non_de = summary.iloc[selected_indices].reset_index(drop=True)
common_non_de.insert(0, 'non_de_rank', np.arange(1, len(common_non_de) + 1))

if set(common_non_de['gene']) & perturbation_target_set:
    raise ValueError('At least one perturbation target entered the PCA feature set.')

common_non_de.to_csv(NON_DE_PATH, index=False)
print(f'Saved {len(common_non_de):,} common non-DE genes to {NON_DE_PATH.name}.')
print('Overlap between PCA genes and analyzed perturbation targets: 0')

# Original notebook computation cell 7
pca_sample_mask = pseudobulk.obs['perturbation'].astype(str).isin(pca_labels).to_numpy()
pca_indices = np.flatnonzero(pca_sample_mask)
pca_obs = pseudobulk.obs.iloc[pca_indices].copy()

if not pca_indices.size:
    raise ValueError('No pseudobulk observations were selected for PCA.')

pca_genes = common_non_de['gene'].astype(str).to_numpy()
pca_gene_indices = expression_gene_lookup.loc[pca_genes].to_numpy(dtype=np.int64)

sort_order = np.argsort(pca_gene_indices)
pca_gene_indices = pca_gene_indices[sort_order]
pca_genes = pca_genes[sort_order]

expression = as_dense(pseudobulk[:, pca_gene_indices].X[pca_indices, :], dtype=np.float32)

feature_mean = expression.mean(axis=0, dtype=np.float64).astype(np.float32)
feature_std = expression.std(axis=0, dtype=np.float64).astype(np.float32)
variable = feature_std > 1e-8

if variable.sum() < N_PCS:
    raise ValueError(f'Only {variable.sum()} variable PCA features remain, fewer than N_PCS={N_PCS}.')

expression -= feature_mean
expression[:, variable] /= feature_std[variable]
expression = expression[:, variable]
pca_genes = pca_genes[variable]

pca = PCA(svd_solver='covariance_eigh').fit(expression)
n_components = pca.explained_variance_ratio_.size
n_saved_components = min(N_SAVED_PCS, n_components)
components = [f'PC{i}' for i in range(1, n_saved_components + 1)]
scores = (expression - pca.mean_) @ pca.components_[:n_saved_components].T

score_table = pd.DataFrame(scores, index=pca_obs.index, columns=components)
score_table.insert(0, 'batch', pca_obs['batch'].astype(str).to_numpy())
score_table.insert(0, 'perturbation', pca_obs['perturbation'].astype(str).to_numpy())
score_table.to_csv(PCA_SCORES_PATH, compression='gzip')

loading_table = pd.DataFrame(
    pca.components_[:n_saved_components].T,
    index=pca_genes,
    columns=components
)
loading_table.index.name = 'gene'
loading_table.to_csv(PCA_LOADINGS_PATH, compression='gzip')

print(f'PCA input: {expression.shape[0]:,} samples × {expression.shape[1]:,} standardized variable non-DE genes.')
print(f'Fitted {n_components:,} components and saved the first {n_saved_components:,}.')

# PCA variance data, without plotting
if N_PCS > n_saved_components:
    raise ValueError(f'Requested {N_PCS} PCs, but only {n_saved_components} were saved.')

eigenvalues = pca.explained_variance_
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
component_number = np.arange(1, len(explained) + 1)
retained_variance = cumulative[N_PCS - 1]
rank_80 = int(np.searchsorted(cumulative, 0.80) + 1)
rank_90 = int(np.searchsorted(cumulative, 0.90) + 1)
first_100_rank = min(N_SAVED_PCS, len(explained))
first_100_variance = cumulative[first_100_rank - 1]

variance_table = pd.DataFrame({'component': component_number, 'eigenvalue': eigenvalues, 'explained_variance_ratio': explained, 'cumulative_explained_variance_ratio': cumulative})
variance_table.to_csv(PCA_VARIANCE_PATH, index=False)



# Original notebook computation cell 15
ranked_genes = summary.sort_values(['average_de_rank', 'mean_abs_z_score', 'mean_abs_logfoldchange', 'gene'], ascending=[True, False, False, True])
global_top20 = ranked_genes.head(TOP_K).reset_index(drop=True)
global_top20.insert(0, 'global_rank', np.arange(1, len(global_top20) + 1))

overlap = global_top20['gene'].astype(str).isin(perturbation_target_set)
removed_genes = global_top20.loc[overlap].reset_index(drop=True)
remaining_genes = ranked_genes.loc[~ranked_genes['gene'].astype(str).isin(perturbation_target_set)].head(N_OUTCOMES).reset_index(drop=True)
if len(remaining_genes) != N_OUTCOMES:
    raise ValueError(f'Only {len(remaining_genes)} eligible outcomes remain; expected {N_OUTCOMES}.')

global_top20.to_csv(GLOBAL_TOP20_PATH, index=False)
remaining_genes.to_csv(OUTCOME_GENES_PATH, index=False)

print(f'Global Top-{TOP_K} genes: {len(global_top20)}')
print(f'Removed perturbation targets: {len(removed_genes)}')
print(f'Retained outcome genes: {len(remaining_genes)}')

# Original notebook computation cell 17
if not score_table.index.equals(pca_obs.index):
    raise ValueError('PCA scores and pseudobulk observations are not aligned.')

missing_pcs = [column for column in PC_COLUMNS if column not in score_table.columns]
if missing_pcs:
    raise KeyError(f'Missing PCs: {missing_pcs}')

model_indices = pca_indices
obs = pca_obs[['perturbation', 'batch']].astype(str).copy()
labels_all = obs['perturbation'].to_numpy()
batches_all = obs['batch'].to_numpy()
W_all = score_table[PC_COLUMNS].to_numpy(dtype=float)
available_labels = set(labels_all)

x_genes = list(dict.fromkeys(map(str, analysis_perturbations)))
y_genes = remaining_genes['gene'].astype(str).drop_duplicates().tolist()

missing_x_labels = sorted(set(x_genes) - available_labels)
missing_x_expression = sorted(set(x_genes) - expression_genes)
missing_y_expression = sorted(set(y_genes) - expression_genes)
if CONTROL not in available_labels:
    raise ValueError('Control observations are missing.')
if missing_x_labels:
    raise ValueError(f'Target perturbation labels are missing: {missing_x_labels[:20]}')
if missing_x_expression:
    raise ValueError(f'Target expression genes are missing: {missing_x_expression[:20]}')
if missing_y_expression:
    raise ValueError(f'Outcome expression genes are missing: {missing_y_expression[:20]}')
if set(x_genes) & set(y_genes):
    raise ValueError('The X and Y gene sets overlap.')
if not y_genes:
    raise ValueError('No outcome genes remain.')
if not np.isfinite(W_all).all():
    raise ValueError('PC scores contain non-finite values.')

model_genes = x_genes + y_genes
model_gene_indices = expression_gene_lookup.loc[model_genes].to_numpy(dtype=np.int64)
model_expression = as_dense(pseudobulk[:, model_gene_indices].X, dtype=float)[model_indices, :]
if not np.isfinite(model_expression).all():
    raise ValueError('Model expression contains non-finite values.')

model_gene_columns = {gene: index for index, gene in enumerate(model_genes)}
y_model_columns = [model_gene_columns[gene] for gene in y_genes]
positions_by_label = {label: np.flatnonzero(labels_all == label) for label in [CONTROL, *x_genes]}

print(f'Target perturbations: {len(x_genes):,}')
print(f'Y genes per perturbation: {len(y_genes):,}')
print(f'Expected perturbation–Y tests: {len(x_genes) * len(y_genes):,}')
print(f'W variables: {N_PCS} PCs')
print(f'Preloaded expression matrix: {model_expression.shape[0]:,} samples × {model_expression.shape[1]:,} required genes')

# Original notebook computation cell 19
def residualize(values, W):
    design = np.column_stack([np.ones(len(W)), W])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def cluster_omega(moment, clusters):
    moment = np.asarray(moment, dtype=float)
    if moment.ndim == 1:
        moment = moment[:, None]
    moment = moment - moment.mean(axis=0)
    codes, labels = pd.factorize(np.asarray(clusters), sort=False)
    if len(labels) < 2:
        raise ValueError('At least two batches are required.')
    cluster_sum = np.zeros((len(labels), moment.shape[1]))
    np.add.at(cluster_sum, codes, moment)
    return (len(labels) / (len(labels) - 1)) * cluster_sum.T @ cluster_sum / len(moment)


def pvalue_normal(beta, se):
    return 2 * norm.sf(abs(beta / se)) if np.isfinite(se) and se > 0 else np.nan


def local_robust_inference(beta, se, sensitivity, radii, n, alpha=ALPHA):
    """Worst-case local-bias CI and p-value for a fixed ellipsoidal moment set."""
    sigma = se * np.sqrt(n)
    sensitivity = np.asarray(sensitivity, dtype=float)
    radii = np.asarray(radii, dtype=float)
    bias_bound = float(np.linalg.norm(sensitivity * radii))
    if not np.isfinite(sigma) or sigma <= 0:
        return bias_bound, np.nan, np.nan, np.nan, np.nan
    t = bias_bound / sigma
    target = 1 - alpha
    critical = brentq(
        lambda c: norm.cdf(c - t) - norm.cdf(-c - t) - target,
        0, t + norm.ppf(1 - alpha / 2) + 10,
    )
    half_width = critical * se
    test_stat = abs(beta / se)
    worst_p = norm.sf(test_stat - t) + norm.cdf(-test_stat - t)
    return bias_bound, critical, beta - half_width, beta + half_width, min(1.0, worst_p)


def fit_one_moment(a, b, clusters):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    b_mean = b.mean()
    if abs(b_mean) < 1e-12:
        raise ValueError('Near-zero identifying moment.')
    beta = a.mean() / b_mean
    omega = cluster_omega(a - beta * b, clusters)[0, 0]
    se = np.sqrt(max(omega / (len(a) * b_mean**2), 0))
    return beta, se, pvalue_normal(beta, se)


def fit_two_step_gmm(a, b, clusters):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a_mean, derivative = a.mean(axis=0), b.mean(axis=0)
    denominator_1 = derivative @ derivative
    if denominator_1 < 1e-12:
        raise ValueError('First-step GMM is unidentified.')

    beta_1 = derivative @ a_mean / denominator_1
    omega_1 = cluster_omega(a - beta_1 * b, clusters)
    weight = np.linalg.pinv(omega_1, rcond=1e-10)
    denominator_2 = derivative @ weight @ derivative
    if denominator_2 < 1e-12:
        raise ValueError('Second-step GMM is unidentified.')

    beta_2 = derivative @ weight @ a_mean / denominator_2
    moment_2 = a - beta_2 * b
    omega_2 = cluster_omega(moment_2, clusters)
    variance = derivative @ weight @ omega_2 @ weight @ derivative / (len(a) * denominator_2**2)
    se = np.sqrt(max(float(variance), 0))
    moment_mean = moment_2.mean(axis=0)
    j_stat = max(float(len(a) * moment_mean @ weight @ moment_mean), 0)
    p_j = chi2.sf(j_stat, df=a.shape[1] - 1)
    return beta_1, beta_2, se, pvalue_normal(beta_2, se), j_stat, p_j, (derivative @ weight / denominator_2), omega_2


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return adjusted
    values = pvalues[valid]
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.clip(ranked, 0, 1)
    adjusted[valid] = result
    return adjusted

# Original notebook computation cell 21
rows, first_stage_rows, skipped = [], [], []

for number, perturbation in enumerate(x_genes, start=1):
    row_positions = np.sort(np.concatenate([positions_by_label[CONTROL], positions_by_label[perturbation]]))

    try:
        used_labels = set(labels_all[row_positions])
        if used_labels != {CONTROL, perturbation}:
            raise ValueError(f'Expected control + {perturbation}, found {sorted(used_labels)}.')

        W = W_all[row_positions]
        batches = batches_all[row_positions]
        z = (labels_all[row_positions] == perturbation).astype(float)
        column_positions = [model_gene_columns[perturbation], *y_model_columns]
        expression_block = model_expression[np.ix_(row_positions, column_positions)]

        centered = expression_block - expression_block.mean(axis=0)
        residualized = residualize(expression_block, W)
        z_centered = z - z.mean()
        x_centered, x_tilde = centered[:, 0], residualized[:, 0]

        fs_beta, fs_se, fs_p = fit_one_moment(z_centered * x_centered, z_centered**2, batches)
        fs_F = (fs_beta / fs_se)**2 if np.isfinite(fs_se) and fs_se > 0 else np.nan
        base = {'perturbation': perturbation, 'x_gene': perturbation, 'n_samples': len(row_positions), 'n_treated': int(z.sum()), 'n_control': int((z == 0).sum()), 'n_batches': np.unique(batches).size}
        first_stage_rows.append({**base, 'first_stage_beta': fs_beta, 'first_stage_se': fs_se, 'first_stage_p': fs_p, 'first_stage_F': fs_F})

    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        skipped.append({'perturbation': perturbation, 'y_gene': None, 'error': str(error)})
        continue

    for y_index, y_gene in enumerate(y_genes, start=1):
        try:
            y_centered, y_tilde = centered[:, y_index], residualized[:, y_index]
            beta_iv, se_iv, p_iv = fit_one_moment(z_centered * y_centered, z_centered * x_centered, batches)
            beta_sv, se_sv, p_sv = fit_one_moment(x_tilde * y_tilde, x_tilde**2, batches)

            gmm_a = np.column_stack([z_centered * y_centered, x_tilde * y_tilde])
            gmm_b = np.column_stack([z_centered * x_centered, x_tilde**2])
            beta_gmm_step1, beta_gmm, se_gmm, p_gmm, j_gmm, p_j_gmm, k_gmm, omega_gmm = fit_two_step_gmm(gmm_a, gmm_b, batches)

            # Use one common moment-deviation set for IV, SV, and GMM on this pair.
            radii = np.array([LOCAL_KAPPA_IV, LOCAL_KAPPA_SV]) * np.sqrt(np.maximum(np.diag(omega_gmm), 0))
            k_iv = np.array([1 / gmm_b[:, 0].mean(), 0.0])
            k_sv = np.array([0.0, 1 / gmm_b[:, 1].mean()])
            robust = {
                method: local_robust_inference(beta, se, k, radii, len(row_positions))
                for method, beta, se, k in (
                    ('iv', beta_iv, se_iv, k_iv),
                    ('sv', beta_sv, se_sv, k_sv),
                    ('gmm', beta_gmm, se_gmm, k_gmm),
                )
            }
            robust_columns = {
                f'{column}_{method}': value
                for method, values in robust.items()
                for column, value in zip(
                    ('local_bias_bound', 'robust_critical', 'ci_low_robust', 'ci_high_robust', 'p_robust'), values
                )
            }

            rows.append({**base, 'y_gene': y_gene, 'beta_iv': beta_iv, 'se_iv': se_iv, 'p_iv': p_iv, 'beta_sv': beta_sv, 'se_sv': se_sv, 'p_sv': p_sv, 'beta_gmm_step1': beta_gmm_step1, 'beta_gmm': beta_gmm, 'se_gmm': se_gmm, 'p_gmm': p_gmm, 'j_stat_gmm': j_gmm, 'p_j_gmm': p_j_gmm, 'local_radius_iv': radii[0], 'local_radius_sv': radii[1], **robust_columns})

        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            skipped.append({'perturbation': perturbation, 'y_gene': y_gene, 'error': str(error)})

    if number % 25 == 0 or number == len(x_genes):
        print(f'Processed {number:,}/{len(x_genes):,} perturbations.')

results = pd.DataFrame(rows)
first_stage = pd.DataFrame(first_stage_rows)
skipped_table = pd.DataFrame(skipped, columns=['perturbation', 'y_gene', 'error'])

# Original notebook computation cell 23
if results.empty:
    if not skipped_table.empty:
        print(skipped_table['error'].value_counts().head(20))
        print(skipped_table.head(20))
    raise RuntimeError('No models were estimated successfully.')

for method in METHODS:
    results[f'q_{method}'] = results.groupby('perturbation', sort=False)[f'p_{method}'].transform(bh_adjust)
    results[f'ci_low_{method}'] = results[f'beta_{method}'] - norm.ppf(1 - ALPHA / 2) * results[f'se_{method}']
    results[f'ci_high_{method}'] = results[f'beta_{method}'] + norm.ppf(1 - ALPHA / 2) * results[f'se_{method}']
    results[f'sig_{method}'] = results[f'q_{method}'].le(ALPHA)
    results[f'q_robust_{method}'] = results.groupby('perturbation', sort=False)[f'p_robust_{method}'].transform(bh_adjust)
    results[f'sig_robust_{method}'] = results[f'q_robust_{method}'].le(ALPHA)

results = results.sort_values(['perturbation', 'y_gene']).reset_index(drop=True)
first_stage = first_stage.sort_values('perturbation').reset_index(drop=True)
tests_per_perturbation = results.groupby('perturbation', sort=False).size()

results.to_csv(RESULTS_PATH, index=False, compression='gzip')
first_stage.to_csv(FIRST_STAGE_PATH, index=False)
skipped_table.to_csv(SKIPPED_PATH, index=False)

print(f'Completed {len(results):,} models; tests per perturbation: {tests_per_perturbation.min():,}–{tests_per_perturbation.max():,}.')
print(f'Saved model results to {RESULTS_PATH.name}, first-stage results to {FIRST_STAGE_PATH.name}, and skipped models to {SKIPPED_PATH.name}.')

# Save matched raw Wilcoxon p-values so the notebook needs no H5AD access.
wilcoxon_perturbation_lookup = pd.Series(np.arange(de.n_obs), index=de.obs['perturbation'].astype(str))
wilcoxon_gene_lookup = pd.Series(np.arange(de.n_vars), index=de.var_names.astype(str))
wilcoxon_rows = wilcoxon_perturbation_lookup.loc[x_genes].to_numpy(dtype=np.int64)
wilcoxon_columns = wilcoxon_gene_lookup.loc[y_genes].to_numpy(dtype=np.int64)
wilcoxon_view = de[:, wilcoxon_columns]
p_raw_block = as_dense(wilcoxon_view.layers[WILCOXON_RAW_P_LAYER], dtype=float)[wilcoxon_rows, :]
q_full_block = as_dense(wilcoxon_view.layers['pvalue_adj'], dtype=float)[wilcoxon_rows, :]
wilcoxon_index = pd.MultiIndex.from_product([x_genes, y_genes], names=['perturbation', 'y_gene'])
matched = pd.DataFrame({'wilcoxon_p_raw': p_raw_block.ravel(), 'wilcoxon_q_all_genes': q_full_block.ravel()}, index=wilcoxon_index)
matched = matched.reindex(pd.MultiIndex.from_frame(results[['perturbation', 'y_gene']]))
if matched.isna().all(axis=1).any():
    raise ValueError('Some model pairs are missing from the matched Wilcoxon table.')
matched.reset_index().to_csv(OUTPUT_DIR / 'wilcoxon_matched_raw.csv.gz', index=False, compression='gzip')
print(f'Saved matched Wilcoxon p-values for {len(matched):,} model pairs.')
pseudobulk.file.close()
de.file.close()
