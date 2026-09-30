"""GO:BP over-representation for Replogle IV/SV/GMM and matched Wilcoxon.

Run after run_replogle.py. Inputs are the official human GAF and go-basic OBO
in data/go; no network access occurs inside this script. The tested outcome
genes, not the whole genome, form the ORA universe.
"""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


METHODS = ("iv", "sv", "gmm")
GO_SOURCES = {
    "gaf": "https://current.geneontology.org/annotations/gaf/HUMAN-uniprot.gaf.gz",
    "obo": "https://current.geneontology.org/ontology/go-basic.obo",
}


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def read_obo(path):
    """Read GO ID, name, namespace, and is_a/part_of parents from go-basic."""
    terms = {}
    entry = None

    def finish():
        if entry and "id" in entry and not entry.get("obsolete"):
            terms[entry["id"]] = entry

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                finish()
                entry = {"parents": []}
            elif line.startswith("["):
                finish()
                entry = None
            elif entry is not None:
                if line.startswith("id: "):
                    entry["id"] = line[4:]
                elif line.startswith("name: "):
                    entry["name"] = line[6:]
                elif line.startswith("namespace: "):
                    entry["namespace"] = line[11:]
                elif line.startswith("is_a: "):
                    entry["parents"].append(line[6:].split(" ! ")[0])
                elif line.startswith("relationship: part_of "):
                    entry["parents"].append(line.split()[2])
                elif line == "is_obsolete: true":
                    entry["obsolete"] = True
        finish()
    return terms


def read_gaf(path, universe):
    """Map human gene symbols in the tested universe to direct BP GO IDs."""
    direct = defaultdict(set)
    header = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                header.append(line.strip())
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15 or fields[8] != "P" or "NOT" in fields[3].split("|"):
                continue
            gene = fields[2]
            if gene in universe:
                direct[gene].add(fields[4])
    return direct, header


def propagated_bp_sets(terms, direct, universe):
    bp = {go for go, item in terms.items() if item.get("namespace") == "biological_process"}
    cache = {}

    def ancestors(go):
        if go in cache:
            return cache[go]
        seen = set()
        pending = [go]
        while pending:
            current = pending.pop()
            if current in seen or current not in bp:
                continue
            seen.add(current)
            pending.extend(terms[current]["parents"])
        cache[go] = seen
        return seen

    gene_sets = defaultdict(set)
    for gene in universe:
        for go in direct.get(gene, ()):
            for ancestor in ancestors(go):
                gene_sets[ancestor].add(gene)
    return gene_sets


def make_calls(results, wilcoxon, alpha):
    if results.duplicated(["perturbation", "y_gene"]).any():
        raise ValueError("Duplicate perturbation-outcome model pairs.")
    if wilcoxon.duplicated(["perturbation", "y_gene"]).any():
        raise ValueError("Duplicate matched Wilcoxon pairs.")
    pair = results.merge(wilcoxon, on=["perturbation", "y_gene"], how="left", validate="one_to_one")
    needed = [f"q_{m}" for m in METHODS] + [f"q_robust_{m}" for m in METHODS] + ["wilcoxon_p_raw"]
    if pair[needed].isna().any().any():
        raise ValueError("GO comparisons require complete IV/SV/GMM and Wilcoxon p/q-values.")
    pair["wilcoxon_q_same_y"] = pair.groupby("perturbation", sort=False)["wilcoxon_p_raw"].transform(bh_adjust)
    for m in METHODS:
        pair[f"wald_{m}"] = pair[f"q_{m}"].le(alpha)
        pair[f"local_{m}"] = pair[f"q_robust_{m}"].le(alpha)
    pair["wilcoxon"] = pair["wilcoxon_q_same_y"].le(alpha)
    return pair


def run(results_path, wilcoxon_path, gaf_path, obo_path, output_dir, alpha=0.05, min_size=2):
    for path in (results_path, wilcoxon_path, gaf_path, obo_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. GO sources: {GO_SOURCES}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair = make_calls(pd.read_csv(results_path), pd.read_csv(wilcoxon_path), alpha)
    universe = set(pair["y_gene"].astype(str))
    if not universe:
        raise ValueError("No tested outcome genes.")
    terms = read_obo(obo_path)
    direct, gaf_header = read_gaf(gaf_path, universe)
    gene_sets = propagated_bp_sets(terms, direct, universe)
    selected_terms = sorted(go for go, genes in gene_sets.items() if min_size <= len(genes) < len(universe))
    if not selected_terms:
        raise ValueError("No GO:BP terms pass the background-size filter.")
    labels = {go: terms[go]["name"] for go in selected_terms}
    sizes = np.array([len(gene_sets[go]) for go in selected_terms])
    modes = tuple(f"{rule}_{m}" for rule in ("wald", "local") for m in METHODS) + ("wilcoxon",)
    rows = []
    selected_rows = []
    for perturbation, frame in pair.groupby("perturbation", sort=False):
        genes_tested = set(frame["y_gene"].astype(str))
        if genes_tested != universe:
            raise ValueError(f"{perturbation}: ORA backgrounds differ across perturbations.")
        for mode in modes:
            hit_genes = set(frame.loc[frame[mode], "y_gene"].astype(str))
            selected_rows.append({"perturbation": perturbation, "selection": mode, "n_tested": len(universe), "n_selected": len(hit_genes), "genes": ";".join(sorted(hit_genes))})
            overlaps = [gene_sets[go] & hit_genes for go in selected_terms]
            counts = np.array([len(overlap) for overlap in overlaps])
            pvalues = hypergeom.sf(counts - 1, len(universe), sizes, len(hit_genes))
            qvalues = bh_adjust(pvalues)
            for go, term_size, overlap, pvalue, qvalue in zip(selected_terms, sizes, overlaps, pvalues, qvalues):
                rows.append({"perturbation": perturbation, "selection": mode, "go_id": go, "go_name": labels[go], "background_size": len(universe), "term_size": int(term_size), "n_selected": len(hit_genes), "overlap": len(overlap), "p_value": float(pvalue), "q_value": float(qvalue), "significant": bool(qvalue <= alpha), "overlap_genes": ";".join(sorted(overlap))})
    enrichment = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    enrichment.to_csv(output_dir / "go_bp_enrichment.csv.gz", index=False, compression="gzip")
    selected.to_csv(output_dir / "go_selected_genes.csv.gz", index=False, compression="gzip")
    significant = enrichment.loc[enrichment["significant"]]
    summary = selected[["perturbation", "selection", "n_selected"]].merge(
        significant.groupby(["perturbation", "selection"]).size().rename("n_significant_terms").reset_index(),
        on=["perturbation", "selection"], how="left", validate="one_to_one",
    )
    summary["n_significant_terms"] = summary["n_significant_terms"].fillna(0).astype(int)
    summary.to_csv(output_dir / "go_bp_per_perturbation.csv", index=False)
    overview = summary.groupby("selection", as_index=False).agg(
        n_perturbations=("perturbation", "size"), selected_pairs=("n_selected", "sum"),
        significant_perturbation_term_pairs=("n_significant_terms", "sum"),
        perturbations_with_go_hits=("n_significant_terms", lambda x: int((x > 0).sum())),
    )
    overview.to_csv(output_dir / "go_bp_overview.csv", index=False)
    recurrence = significant.groupby(["selection", "go_id", "go_name"], as_index=False).agg(
        n_perturbations=("perturbation", "nunique"), minimum_q=("q_value", "min")
    )
    recurrence.to_csv(output_dir / "go_bp_recurrent_terms.csv", index=False)
    manifest = {
        "gaf_url": GO_SOURCES["gaf"], "obo_url": GO_SOURCES["obo"],
        "gaf_sha256": hashlib.sha256(gaf_path.read_bytes()).hexdigest(),
        "obo_sha256": hashlib.sha256(obo_path.read_bytes()).hexdigest(),
        "gaf_header": gaf_header[:25], "alpha": alpha, "min_term_size": min_size,
        "n_tested_outcome_genes": len(universe), "n_genes_with_direct_bp_annotation": len(direct),
        "n_tested_bp_terms": len(selected_terms),
        "annotation_policy": "human GAF, aspect P, exclude NOT; propagate is_a and part_of; background=all tested outcomes",
    }
    (output_dir / "go_bp_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(overview.to_string(index=False))
    print(f"GO:BP terms tested per perturbation/selection: {len(selected_terms)}; results: {output_dir}")
    return enrichment, overview


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results/replogle/iv_sv_gmm_results.csv.gz")
    parser.add_argument("--wilcoxon", type=Path, default=root / "results/replogle/wilcoxon_matched_raw.csv.gz")
    parser.add_argument("--gaf", type=Path, default=root / "data/go/HUMAN-uniprot.gaf.gz")
    parser.add_argument("--obo", type=Path, default=root / "data/go/go-basic.obo")
    parser.add_argument("--output-dir", type=Path, default=root / "results/replogle")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-term-size", type=int, default=2)
    args = parser.parse_args()
    run(args.results, args.wilcoxon, args.gaf, args.obo, args.output_dir, args.alpha, args.min_term_size)


if __name__ == "__main__":
    main()
