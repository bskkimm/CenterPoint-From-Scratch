import pytest
import torch

from centerpoint.models.backbones import SparseBackbone, SparseBackboneInput


class FakeSparseBackbone(SparseBackbone):
    def forward_sparse(self, inputs):
        return inputs.features.new_zeros(
            (
                inputs.batch_size,
                self.output_channels,
                (inputs.spatial_shape[1] + self.output_stride - 1) // self.output_stride,
                (inputs.spatial_shape[2] + self.output_stride - 1) // self.output_stride,
            )
        )


class WrongShapeBackbone(SparseBackbone):
    def forward_sparse(self, inputs):
        return inputs.features.new_zeros((inputs.batch_size, self.output_channels, 1, 1))


def make_inputs():
    return SparseBackboneInput(
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        coordinates=torch.tensor([[0, 1, 2, 3], [1, 3, 15, 23]], dtype=torch.int32),
        spatial_shape=(4, 16, 24),
        batch_size=2,
    )


def test_sparse_backbone_validates_dense_bev_output():
    output = FakeSparseBackbone(output_channels=4, output_stride=8)(make_inputs())

    assert output.shape == (2, 4, 2, 3)
    assert output.dtype == torch.float32


def test_sparse_input_rejects_xyz_or_out_of_bounds_coordinates():
    with pytest.raises(ValueError, match="outside"):
        SparseBackboneInput(
            features=torch.ones((1, 2)),
            coordinates=torch.tensor([[0, 4, 0, 0]], dtype=torch.int32),
            spatial_shape=(4, 16, 24),
            batch_size=1,
        )
    with pytest.raises(ValueError, match="outside"):
        SparseBackboneInput(
            features=torch.ones((1, 2)),
            coordinates=torch.tensor([[1, 0, 0, 0]], dtype=torch.int32),
            spatial_shape=(4, 16, 24),
            batch_size=1,
        )


def test_sparse_backbone_rejects_backend_shape_drift():
    with pytest.raises(ValueError, match="dense BEV shape"):
        WrongShapeBackbone(output_channels=4, output_stride=8)(make_inputs())


def test_sparse_input_supports_empty_voxel_sets():
    inputs = SparseBackboneInput(
        features=torch.empty((0, 5)),
        coordinates=torch.empty((0, 4), dtype=torch.int32),
        spatial_shape=(4, 16, 24),
        batch_size=1,
    )

    assert FakeSparseBackbone(output_channels=2, output_stride=8)(inputs).shape == (1, 2, 2, 3)


def test_sparse_backbone_uses_official_ceil_geometry_for_odd_grids():
    inputs = SparseBackboneInput(
        features=torch.ones((1, 2)),
        coordinates=torch.tensor([[0, 0, 14, 14]], dtype=torch.int32),
        spatial_shape=(4, 15, 15),
        batch_size=1,
    )

    assert FakeSparseBackbone(output_channels=2, output_stride=8)(inputs).shape == (1, 2, 2, 2)
