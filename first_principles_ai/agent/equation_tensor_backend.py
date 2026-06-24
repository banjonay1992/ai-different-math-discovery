from __future__ import annotations

"""Optional tensor kernels for equation scoring.

The workbench still generates candidate equations from primitive observations.
This module only accelerates numeric reductions used to score those candidates.
"""

from typing import Any


def available_equation_scoring_backends() -> dict[str, Any]:
    report = {
        'numpy': False,
        'torch': False,
        'cuda': False,
        'cuda_device': None,
    }
    try:
        import numpy  # noqa: F401

        report['numpy'] = True
    except Exception:
        pass
    try:
        import torch  # type: ignore

        report['torch'] = True
        report['cuda'] = bool(torch.cuda.is_available())
        if report['cuda']:
            report['cuda_device'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return report


def resolve_equation_scoring_backend(requested: str | None) -> dict[str, Any]:
    requested = (requested or 'python').lower().replace('-', '_')
    available = available_equation_scoring_backends()
    if requested in {'none', 'python'}:
        backend = 'python'
        reason = None
    elif requested in {'auto', 'tensor'}:
        if available['cuda']:
            backend = 'cuda'
        elif available['torch']:
            backend = 'torch'
        elif available['numpy']:
            backend = 'numpy'
        else:
            backend = 'python'
        reason = None if backend != 'python' else 'no_tensor_backend_available'
    elif requested == 'cuda':
        if available['cuda']:
            backend = 'cuda'
            reason = None
        elif available['torch']:
            backend = 'torch'
            reason = 'cuda_unavailable_using_torch_cpu'
        elif available['numpy']:
            backend = 'numpy'
            reason = 'cuda_unavailable_using_numpy'
        else:
            backend = 'python'
            reason = 'cuda_unavailable_no_tensor_backend'
    elif requested == 'torch':
        if available['torch']:
            backend = 'torch'
            reason = None
        elif available['numpy']:
            backend = 'numpy'
            reason = 'torch_unavailable_using_numpy'
        else:
            backend = 'python'
            reason = 'torch_unavailable_no_tensor_backend'
    elif requested == 'numpy':
        backend = 'numpy' if available['numpy'] else 'python'
        reason = None if backend == 'numpy' else 'numpy_unavailable'
    else:
        raise ValueError(f"Unknown equation scoring backend: {requested}")
    return {
        'requested_backend': requested,
        'backend': backend,
        'accelerated': backend != 'python',
        'fallback_reason': reason,
        'available_backends': available,
    }


def fitted_scale_reduce(
    features: list[float],
    targets: list[float],
    *,
    backend: str,
) -> float | None:
    if backend == 'python':
        numerator = sum(target * feature for target, feature in zip(targets, features))
        denominator = sum(feature * feature for feature in features)
        if abs(denominator) < 1e-12:
            return None
        return numerator / denominator
    if backend in {'torch', 'cuda'}:
        try:
            return _torch_fitted_scale(features, targets, backend=backend)
        except Exception:
            if backend == 'cuda':
                return _torch_fitted_scale(features, targets, backend='torch')
            raise
    return _numpy_fitted_scale(features, targets)


def vector_scale_reduce(
    x_features: list[float],
    y_features: list[float],
    x_targets: list[float],
    y_targets: list[float],
    *,
    backend: str,
) -> float | None:
    if backend == 'python':
        numerator = sum(
            tx * fx + ty * fy
            for fx, fy, tx, ty in zip(x_features, y_features, x_targets, y_targets)
        )
        denominator = sum(
            fx * fx + fy * fy
            for fx, fy in zip(x_features, y_features)
        )
        if abs(denominator) < 1e-12:
            return None
        return numerator / denominator
    if backend in {'torch', 'cuda'}:
        try:
            return _torch_vector_scale(
                x_features,
                y_features,
                x_targets,
                y_targets,
                backend=backend,
            )
        except Exception:
            if backend == 'cuda':
                return _torch_vector_scale(
                    x_features,
                    y_features,
                    x_targets,
                    y_targets,
                    backend='torch',
                )
            raise
    return _numpy_vector_scale(x_features, y_features, x_targets, y_targets)


def _numpy_fitted_scale(features: list[float], targets: list[float]) -> float | None:
    import numpy as np

    feature_values = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    denominator = float(np.dot(feature_values, feature_values))
    if abs(denominator) < 1e-12:
        return None
    numerator = float(np.dot(target_values, feature_values))
    return numerator / denominator


def _numpy_vector_scale(
    x_features: list[float],
    y_features: list[float],
    x_targets: list[float],
    y_targets: list[float],
) -> float | None:
    import numpy as np

    fx = np.asarray(x_features, dtype=np.float64)
    fy = np.asarray(y_features, dtype=np.float64)
    tx = np.asarray(x_targets, dtype=np.float64)
    ty = np.asarray(y_targets, dtype=np.float64)
    denominator = float(np.dot(fx, fx) + np.dot(fy, fy))
    if abs(denominator) < 1e-12:
        return None
    numerator = float(np.dot(tx, fx) + np.dot(ty, fy))
    return numerator / denominator


def _torch_fitted_scale(
    features: list[float],
    targets: list[float],
    *,
    backend: str,
) -> float | None:
    import torch  # type: ignore

    device = torch.device('cuda' if backend == 'cuda' and torch.cuda.is_available() else 'cpu')
    feature_values = torch.tensor(features, dtype=torch.float64, device=device)
    target_values = torch.tensor(targets, dtype=torch.float64, device=device)
    denominator = torch.dot(feature_values, feature_values)
    denominator_value = float(denominator.detach().cpu().item())
    if abs(denominator_value) < 1e-12:
        return None
    numerator = torch.dot(target_values, feature_values)
    return float((numerator / denominator).detach().cpu().item())


def _torch_vector_scale(
    x_features: list[float],
    y_features: list[float],
    x_targets: list[float],
    y_targets: list[float],
    *,
    backend: str,
) -> float | None:
    import torch  # type: ignore

    device = torch.device('cuda' if backend == 'cuda' and torch.cuda.is_available() else 'cpu')
    fx = torch.tensor(x_features, dtype=torch.float64, device=device)
    fy = torch.tensor(y_features, dtype=torch.float64, device=device)
    tx = torch.tensor(x_targets, dtype=torch.float64, device=device)
    ty = torch.tensor(y_targets, dtype=torch.float64, device=device)
    denominator = torch.dot(fx, fx) + torch.dot(fy, fy)
    denominator_value = float(denominator.detach().cpu().item())
    if abs(denominator_value) < 1e-12:
        return None
    numerator = torch.dot(tx, fx) + torch.dot(ty, fy)
    return float((numerator / denominator).detach().cpu().item())
