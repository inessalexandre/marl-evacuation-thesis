import warnings

from pettingzoo.mpe.simple_crypto.simple_crypto import env, parallel_env, raw_env

warnings.warn(
    "The environment `pettingzoo.mpe.simple_evacuation_v0` has been moved to `mpe2.simple_evacuation_v0` and will be removed in a future release. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["env", "parallel_env", "raw_env"]
