import gzip

import numpy as np

from experiments.replogle.run_go_analysis import (
    bh_adjust, propagated_bp_sets, read_gaf, read_obo,
)


def test_go_parser_excludes_not_and_propagates_bp_parent(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text(
        "[Term]\nid: GO:0000001\nname: parent\nnamespace: biological_process\n\n"
        "[Term]\nid: GO:0000002\nname: child\nnamespace: biological_process\n"
        "is_a: GO:0000001 ! parent\n\n"
        "[Term]\nid: GO:0000003\nname: molecular function\n"
        "namespace: molecular_function\n"
    )
    gaf = tmp_path / "human.gaf.gz"
    def record(gene, qualifier, go, aspect):
        fields = ["UniProtKB", gene, gene, qualifier, go, "PMID:1", "EXP", "", aspect]
        return "\t".join(fields + ["x"] * 8) + "\n"
    with gzip.open(gaf, "wt") as handle:
        handle.write("!gaf-version: 2.2\n")
        handle.write(record("A", "", "GO:0000002", "P"))
        handle.write(record("B", "NOT", "GO:0000002", "P"))
        handle.write(record("C", "", "GO:0000003", "F"))
    terms = read_obo(obo)
    direct, _ = read_gaf(gaf, {"A", "B", "C"})
    gene_sets = propagated_bp_sets(terms, direct, {"A", "B", "C"})
    assert gene_sets["GO:0000001"] == {"A"}
    assert gene_sets["GO:0000002"] == {"A"}
    assert "GO:0000003" not in gene_sets


def test_bh_adjust_is_monotone_and_bounded():
    adjusted = bh_adjust([0.04, 0.001, 0.3, 1.0])
    np.testing.assert_allclose(adjusted, [0.08, 0.004, 0.4, 1.0])
