"""Small numerical checks, run during image construction and restricted execution."""

import json
from importlib.metadata import version


def check():
    import control
    import cvxpy as cp
    import numpy as np
    import torch
    from scipy.linalg import solve_discrete_are
    from sklearn.linear_model import Ridge

    a, b, q, r = (np.array([[v]]) for v in (1.0, 1.0, 1.0, 1.0))
    p = solve_discrete_are(a, b, q, r)
    gain, _, poles = control.dlqr(a, b, q, r, method="scipy")
    np.testing.assert_allclose(gain, np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a))
    assert np.all(np.abs(poles) < 1)

    assert {"OSQP", "CLARABEL", "SCS"} <= set(cp.installed_solvers())
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.square(x - 2)), [x <= 1])
    for solver in ("OSQP", "CLARABEL", "SCS"):
        problem.solve(solver=solver)
        assert problem.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, 1, atol=1e-4)

    model = Ridge(alpha=1e-6).fit([[0], [1], [2]], [0, 2, 4])
    np.testing.assert_allclose(model.predict([[3]]), [6], atol=1e-4)

    # No GPU build is acceptable even when a machine has no GPU attached.
    assert torch.version.cuda is None and torch.version.hip is None
    torch.set_num_threads(1)
    tensor = torch.tensor([1.0, 2.0], requires_grad=True, device="cpu")
    tensor.square().sum().backward()
    np.testing.assert_allclose(tensor.grad.numpy(), [2, 4])
    assert torch.nn.Linear(2, 1)(tensor).device.type == "cpu"
    return {
        name: version(name)
        for name in ("numpy", "scipy", "control", "cvxpy", "scikit-learn", "torch")
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
