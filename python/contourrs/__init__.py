"""contourrs — fast raster polygonization with Arrow export."""

from contourrs._contourrs import contours, contours_arrow, shapes, shapes_arrow

try:
    from contourrs._contourrs import shapes_cuda_raw as _shapes_cuda_raw
except ImportError:
    _shapes_cuda_raw = None


def shapes_cuda(
    source,
    mask=None,
    connectivity=4,
    transform=None,
    device_ordinal=0,
):
    """Polygonize a CUDA tensor (torch.int32) using GPU CCL + CPU tracing."""
    if _shapes_cuda_raw is None:
        raise RuntimeError("CUDA support is not enabled in this build")

    import torch

    if not torch.is_tensor(source):
        source = torch.utils.dlpack.from_dlpack(source)
    if source.dtype != torch.int32:
        raise TypeError("source dtype must be torch.int32")

    if mask is not None and not torch.is_tensor(mask):
        mask = torch.utils.dlpack.from_dlpack(mask)

    return _shapes_cuda_raw(
        source,
        mask=mask,
        connectivity=connectivity,
        transform=transform,
        device_ordinal=device_ordinal,
    )

__all__ = ["contours", "contours_arrow", "shapes", "shapes_arrow", "shapes_cuda"]
