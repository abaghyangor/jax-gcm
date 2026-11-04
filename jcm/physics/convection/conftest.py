"""Local pytest configuration for the convection tests.

The plume-driver tests compare against ModelE's float64 dumps at tolerances
float32 rounding breaks, so they need ``jax_enable_x64``. They cannot simply
turn it on themselves: the root conftest holds the flag at the session default
around *every* test (``_pin_jax_x64``, #729), which runs after a module's
``setUpModule`` and so undoes it before the test body. Depending on that
fixture here orders this one after it, so the setting survives into the test.

Scoped to the one module that needs it -- the rest of the convection suite
asserts the model's float32 default and must keep it.
"""
import pytest

_NEEDS_X64 = "giss_plume_driver_test.py"


@pytest.fixture(autouse=True)
def _plume_driver_needs_x64(request, _pin_jax_x64):
    """Run the plume-driver module's tests in float64."""
    if _NEEDS_X64 not in str(request.node.fspath):
        yield
        return
    import jax
    before = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", before)
