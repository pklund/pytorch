# Owner(s): ["module: inductor"]
"""Tests for the Helion backend in PyTorch Inductor.

This file reuses the full test_torchinductor test suite with config patches
(following the same pattern as test_pallas.py), plus targeted manual tests.
"""

import os
import sys
import unittest

# Set HELION_AUTOTUNE_EFFORT=none early so tests run fast
os.environ["HELION_AUTOTUNE_EFFORT"] = "none"

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_HELION


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

if not HAS_CUDA_AND_TRITON:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires CUDA and Triton")

if not HAS_HELION:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires helion")


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


# Load helion expected failures from sentinel files
_helion_expected_failures_dir = os.path.join(
    os.path.dirname(__file__), "helion_expected_failures"
)
if os.path.isdir(_helion_expected_failures_dir):
    HELION_EXPECTED_FAILURES = set(os.listdir(_helion_expected_failures_dir))
else:
    HELION_EXPECTED_FAILURES = set()

# Load helion skip tests from sentinel files (for flaky tests)
_helion_skip_tests_dir = os.path.join(os.path.dirname(__file__), "helion_skip_tests")
if os.path.isdir(_helion_skip_tests_dir):
    HELION_SKIP_TESTS = set(os.listdir(_helion_skip_tests_dir))
else:
    HELION_SKIP_TESTS = set()


test_classes = {}


def make_helion(cls):
    """Create a test class variant that uses the Helion backend.

    Args:
        cls: The test class to create a Helion variant of.
    """
    patches = [
        (config, "cuda_backend", "helion"),
    ]
    cls_prefix = "Helion"
    suffix = "_helion"

    # Mark tests based on sentinel files in helion_expected_failures/ and helion_skip_tests/
    for name in cls.__dict__:
        if name.startswith("test_"):
            fn = cls.__dict__[name]
            if callable(fn):
                key = f"{cls.__name__}.{name}"
                if key in HELION_EXPECTED_FAILURES:
                    fn._expected_failure_helion = True
                elif key in HELION_SKIP_TESTS:
                    fn._skip_helion = True

    def skip_decorator(fn):
        if hasattr(fn, "_skip_helion"):
            return unittest.skip("Skipped in Helion backend")(fn)
        return fn

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        *patches,
        xfail_prop="_expected_failure_helion",
        decorator=skip_decorator,
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


# Apply to GPU test suites
if test_torchinductor.RUN_GPU:
    make_helion(test_torchinductor.SweepInputsGPUTest)
    make_helion(test_torchinductor.GPUTests)


# --- Manual targeted tests (kept for fast sanity checks) ---


@inductor_config.patch(cuda_backend="helion")
class TestHelionBackend(TestCase):
    device = "cuda"

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_simple_add(self):
        def fn(x, y):
            return x + y

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_simple_mul(self):
        def fn(x, y):
            return x * y

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_pointwise_chain(self):
        def fn(x, y, z):
            return x + y * z

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        z = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y, z), fn(x, y, z), rtol=1e-4, atol=1e-4
        )

    def test_unary_sin(self):
        def fn(x):
            return torch.sin(x)

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_unary_exp(self):
        def fn(x):
            return torch.exp(x)

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_unary_cos(self):
        def fn(x):
            return torch.cos(x)

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_relu(self):
        def fn(x):
            return torch.relu(x)

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_gelu(self):
        def fn(x):
            return torch.nn.functional.gelu(x)

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_fused_ops(self):
        def fn(x, y, z):
            return torch.sin(x) + torch.cos(y) * z

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        z = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y, z), fn(x, y, z), rtol=1e-4, atol=1e-4
        )

    def test_scalar_mul(self):
        def fn(x):
            return x * 2.0

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_2d_tensor(self):
        def fn(x, y):
            return x + y

        x = torch.randn(64, 128, device="cuda")
        y = torch.randn(64, 128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_different_dtypes(self):
        def fn(x, y):
            return x + y

        x = torch.randn(1024, device="cuda", dtype=torch.float16)
        y = torch.randn(1024, device="cuda", dtype=torch.float16)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-3, atol=1e-3)

    def test_where(self):
        def fn(x):
            return torch.where(x > 0, x, torch.zeros_like(x))

        x = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_multiple_outputs(self):
        def fn(x, y):
            return x + y, x * y

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        compiled_fn = torch.compile(fn)
        result = compiled_fn(x, y)
        expected = fn(x, y)
        self.assertEqual(len(result), len(expected))
        torch.testing.assert_close(result[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_3d_tensor(self):
        def fn(x, y):
            return x + y

        x = torch.randn(4, 32, 64, device="cuda")
        y = torch.randn(4, 32, 64, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_broadcasting(self):
        def fn(x, bias):
            return x + bias

        x = torch.randn(64, 128, device="cuda")
        bias = torch.randn(128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, bias), fn(x, bias), rtol=1e-4, atol=1e-4
        )

    def test_sum_reduction(self):
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_max_reduction(self):
        def fn(x):
            return torch.amax(x, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_softmax(self):
        def fn(x):
            return torch.softmax(x, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_layer_norm(self):
        def fn(x):
            return torch.nn.functional.layer_norm(x, [128])

        x = torch.randn(64, 128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_mean_reduction(self):
        def fn(x):
            return torch.mean(x, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    run_tests()
