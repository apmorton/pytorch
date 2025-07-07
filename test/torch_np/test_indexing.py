# Owner(s): ["module: dynamo"]

import numpy

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_array_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_array_equal


@instantiate_parametrized_tests
class TestAdvancedIndexing(TestCase):
    """Test advanced indexing for NumPy compatibility (separated advanced indices)."""

    def _test_pattern(self, shape, cases, name=""):
        """Unified test for getitem/setitem with shape and cases."""
        torch_arr, numpy_arr = np.arange(numpy.prod(shape)).reshape(
            shape
        ), numpy.arange(numpy.prod(shape)).reshape(shape)

        for i, case in enumerate(cases):
            index = (
                case
                if isinstance(case, tuple) and isinstance(case[0], (slice, list, int))
                else case[0]
            )
            value = (
                999 + i
                if len(case) == 1
                or not isinstance(case, tuple)
                or isinstance(case[0], (slice, list, int))
                else case[1]
            )

            with self.subTest(index=index):
                # Test getitem
                tr, nr = torch_arr[index], numpy_arr[index]
                self.assertEqual(tr.shape, nr.shape, f"{name} getitem shape mismatch")
                assert_array_equal(
                    tr.tensor.numpy() if hasattr(tr, "tensor") else tr, nr
                )

                # Test setitem
                tc, nc = torch_arr.copy(), numpy_arr.copy()
                val_np = value.tensor.numpy() if hasattr(value, "tensor") else value
                tc[index], nc[index] = value, val_np
                assert_array_equal(
                    tc.tensor.numpy() if hasattr(tc, "tensor") else tc,
                    nc,
                    f"{name} setitem mismatch",
                )

    def test_basic_patterns(self):
        """Test fundamental indexing patterns across dimensions."""
        test_data = [
            # (shape, test_cases, description)
            ((10,), [([0, 2, 4],), ([1, 3, 5, 7],), ([0, 0, 1, 1],)], "1D"),
            (
                (4, 6),
                [
                    (slice(None), [0]),
                    ([0], slice(None)),
                    (slice(1, 3), [0, 2]),
                    ([0, 2], slice(1, 4)),
                    ([0, 2], 1),
                    (1, [0, 2, 4]),
                ],
                "2D",
            ),
            (
                (3, 4, 5),
                [
                    ([0, 2], slice(None), slice(None)),
                    (slice(None), [0, 2], slice(None)),
                    (slice(None), slice(None), [0, 2, 4]),
                    ([0, 1], slice(None), 2),
                    ([0, 2], 1, slice(None)),
                    (1, [1, 3], slice(None)),
                    (slice(None), [0, 2], 3),
                    ([0, 1], slice(None), [0, 2]),
                    ([1, 2], slice(1, 3), [1, 3]),
                    (slice(None), [0], slice(None)),
                ],
                "3D",
            ),
            (
                (2, 3, 4, 5),
                [
                    (slice(None), [0], 0, slice(None)),
                    (slice(None), [0], slice(None), 0),
                    (slice(None), [0, 1], 0, slice(None)),
                    (slice(None), [0, 1], slice(None), 0),
                    ([0, 1], slice(None), [0, 2], slice(None)),
                    (slice(None), [0, 2], slice(None), [0, 3]),
                    ([0], slice(None), slice(None), [1, 3]),
                    (slice(None), slice(None), [0, 2], [1, 4]),
                ],
                "4D",
            ),
        ]

        for shape, cases, desc in test_data:
            self._test_pattern(shape, cases, desc)

    def test_separated_indices(self):
        """Test multiple separated advanced indices and special patterns."""
        cases = [
            # Multiple separated on different shapes
            (
                (3, 4, 5),
                [
                    ([0, 1], slice(None), [1, 2]),
                    ([0, 2], slice(1, 3), [0, 1]),
                    ([0, 1], 1, [1, 2]),
                ],
            ),
            (
                (2, 3, 4, 5, 2),
                [
                    ([0], slice(None), [1], slice(None), [0]),
                    ([0, 1], 0, [1], slice(None), [0, 1]),
                ],
            ),
            # Adjacent vs separated comparison
            ((3, 4, 5), [([0, 1], [1, 2]), ([0, 1], slice(None), [1, 2])]),
            # Corner cases
            (
                (3, 4, 5),
                [
                    ([0], slice(None), [1]),
                    ([-1], slice(None), [-1]),
                    ([0, -1], slice(None), [1, -1]),
                ],
            ),
        ]

        for shape, test_cases in cases:
            self._test_pattern(shape, test_cases, f"Separated-{shape}")

    def test_special_cases(self):
        """Test edge cases and array setitem."""
        # Edge cases
        edge_cases = [
            ([0, 0, 1, 1], slice(None)), 
            (slice(None), [0, 2, 0, 2]),
            (slice(None), [])  # Empty list indexing
        ]
        self._test_pattern((4, 6), edge_cases, "Edge")

        # Boolean indexing
        torch_arr, numpy_arr = np.arange(24).reshape(4, 6), numpy.arange(24).reshape(
            4, 6
        )
        mask_t, mask_n = torch_arr > 10, numpy_arr > 10
        tr, nr = torch_arr[mask_t], numpy_arr[mask_n]
        self.assertEqual(tr.shape, nr.shape)
        assert_array_equal(tr.tensor.numpy() if hasattr(tr, "tensor") else tr, nr)

        # Array setitem with specific shapes
        torch_arr, numpy_arr = np.arange(420).reshape(3, 5, 7, 4), numpy.arange(
            420
        ).reshape(3, 5, 7, 4)
        specific_cases = [
            ((slice(None), [0], slice(None), 0), (1, 3, 7), (1, 3, 7)),
            ((slice(None), [0, 1], slice(None), 0), (2, 3, 7), (2, 3, 7)),
        ]

        for idx, expected_shape, val_shape in specific_cases:
            # Test specific expected shapes
            tr, nr = torch_arr[idx], numpy_arr[idx]
            self.assertEqual(tr.shape, expected_shape, f"Expected {expected_shape}")
            self.assertEqual(tr.shape, nr.shape, "NumPy compatibility")
            assert_array_equal(tr.tensor.numpy() if hasattr(tr, "tensor") else tr, nr)

            # Test array setitem
            tc, nc = torch_arr.copy(), numpy_arr.copy()
            val = np.ones(val_shape) * 777
            tc[idx], nc[idx] = (
                val,
                val.tensor.numpy() if hasattr(val, "tensor") else val,
            )
            assert_array_equal(tc.tensor.numpy() if hasattr(tc, "tensor") else tc, nc)


if __name__ == "__main__":
    run_tests()
