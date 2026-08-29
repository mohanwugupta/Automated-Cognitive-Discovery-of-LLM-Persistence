from cognitive_discovery.models.flexible.neural import fit_flexible_model
from cognitive_discovery.models.synthetic import generate_teacher_data


def test_mlp_reaches_simple_teacher_ceiling():
    train, test = generate_teacher_data(
        "dynamic_reevaluation", n_train=600, n_test=180, seed=20, noise=0.01
    )
    fit = fit_flexible_model(
        train,
        test,
        kind="mlp",
        seed=20,
        hidden_size=24,
        max_epochs=160,
        patience=20,
    )
    assert fit.metrics["r2"] > 0.90


def test_gru_reaches_simple_sequential_teacher_ceiling():
    train, test = generate_teacher_data(
        "dual_history", n_train=420, n_test=140, seed=28, noise=0.01
    )
    fit = fit_flexible_model(
        train,
        test,
        kind="gru",
        seed=28,
        hidden_size=24,
        max_epochs=120,
        patience=15,
    )
    assert fit.metrics["r2"] > 0.90
