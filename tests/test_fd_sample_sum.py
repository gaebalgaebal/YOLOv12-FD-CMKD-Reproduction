from __future__ import annotations

import unittest

import torch

from ultralytics.utils.fd_cmkd_loss import FD_NLKD_Loss


class FDLossReductionTests(unittest.TestCase):
    def test_sample_sum_is_element_count_times_mean(self) -> None:
        student = torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.0], [1.0, 2.0]]],
                [[[0.5, 1.5], [2.5, 3.5]], [[-0.5, 0.5], [1.5, 2.5]]],
            ]
        )
        teacher = torch.zeros_like(student)
        mean_block = FD_NLKD_Loss(in_channels=2, loss_reduction="mean")
        sum_block = FD_NLKD_Loss(in_channels=2, loss_reduction="sample_sum")

        mean_loss = mean_block._distance_loss(student, teacher)
        sample_sum_loss = sum_block._distance_loss(student, teacher)

        elements_per_sample = student[0].numel()
        torch.testing.assert_close(sample_sum_loss, mean_loss * elements_per_sample)

    def test_default_preserves_upstream_mean(self) -> None:
        block = FD_NLKD_Loss(in_channels=2)
        self.assertEqual(block.loss_reduction, "mean")

    def test_unknown_reduction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported FD loss reduction"):
            FD_NLKD_Loss(in_channels=2, loss_reduction="unknown")


if __name__ == "__main__":
    unittest.main()
