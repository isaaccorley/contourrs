"""contourrs — fast raster polygonization with Arrow export."""

from contourrs._contourrs import shapes

__all__ = ["shapes"]

try:
    from contourrs._contourrs import shapes_arrow

    __all__.append("shapes_arrow")
except ImportError:
    pass
