import pandas as pd

from experiments.run import (
    active_replay_experiments,
    load_inputs,
    representation_experiments,
    svm_experiments,
    write_run_info,
)


def test_analises_adicionais_reproduzem_as_tabelas_da_v2(tmp_path):
    data = load_inputs()
    representation = representation_experiments(data, tmp_path)
    replay = active_replay_experiments(data, tmp_path)
    svm = svm_experiments(data, tmp_path)
    write_run_info(tmp_path)

    hits = representation.set_index("configuration").hits_at_100
    assert hits["PCA-5 + GP"] == 13
    assert hits["PCA-10 + GP"] == 32
    assert hits["PCA-20 + GP"] == 44
    assert hits["PCA-30 + GP"] == 43
    assert hits["PCA-30 whitening + GP"] == 44
    assert hits["Isolation Forest PCA-30"] == 4

    replay_hits = replay.set_index("configuration").hits_at_100
    assert replay_hits["PCA-20 epsilon-0"] == 36
    assert replay_hits["PCA-20 epsilon-3"] == 38
    assert replay_hits["PCA-30 epsilon-0"] == 40
    assert replay_hits["PCA-30 epsilon-3"] == 43
    assert bool(replay.set_index("configuration").loc["PCA-30 epsilon-3", "matches_frozen_top100"])

    svm_hits = svm.set_index("configuration").hits_at_100
    assert svm_hits["Balanced RBF SVM PCA-20"] == 45
    assert svm_hits["Balanced RBF SVM PCA-30"] == 42
    assert pd.read_csv(tmp_path / "pca_variance.csv").components.tolist() == [5, 10, 20, 30]
    assert (tmp_path / "SHA256SUMS").is_file()
