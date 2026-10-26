"""Shared dtype dispatch and Arrow C Data Interface ownership regressions."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import shapely
from contourrs import _contourrs, contours, contours_arrow, shapes, shapes_arrow

DTYPES = ["uint8", "uint16", "uint32", "int16", "int32", "float32", "float64"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", ["shapes", "contours"])
@pytest.mark.parametrize("masked", [False, True])
def test_output_formats_agree_for_every_dtype(dtype, operation, masked):
    source = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]], dtype=dtype)
    mask = source != 2 if masked else None
    kwargs = {"mask": mask, "transform": (10, 2, 500000, 1, -10, 4500000)}
    if operation == "shapes":
        geojson = shapes(source, connectivity=8, **kwargs)
        table = shapes_arrow(source, connectivity=8, **kwargs)
    else:
        geojson = contours(source, [0.5, 1.5, 2.5, 3.5], **kwargs)
        table = contours_arrow(source, [0.5, 1.5, 2.5, 3.5], **kwargs)
    assert table["value"].to_pylist() == [value for _, value in geojson]
    for (geometry, _), encoded in zip(
        geojson, table["geometry"].to_pylist(), strict=True
    ):
        assert shapely.equals_exact(
            shapely.geometry.shape(geometry), shapely.from_wkb(encoded), tolerance=0
        )


@pytest.mark.parametrize(
    "name", ["shapes", "shapes_arrow", "contours", "contours_arrow"]
)
@pytest.mark.parametrize(
    ("source", "error"),
    [
        (np.ones((2, 2), dtype=np.int64), "Unsupported dtype: int64"),
        (np.ones(4, dtype=np.uint8), "Cannot interpret source as uint8"),
    ],
)
def test_native_dispatch_preserves_input_errors(name, source, error):
    function = getattr(_contourrs, name)
    kwargs = {"thresholds": [0, 2]} if name.startswith("contours") else {}
    with pytest.raises(TypeError, match=error):
        function(source, **kwargs)


@pytest.mark.parametrize("operation", ["shapes", "contours"])
@pytest.mark.parametrize("consume", [False, True])
def test_arrow_import_failure_preserves_ownership(monkeypatch, operation, consume):
    source = np.ones((3, 3), dtype=np.uint8)
    imported = []
    # Exercise the same C Data Interface entry point used by the Rust binding.
    import_from_c = pa.RecordBatch._import_from_c  # noqa: SLF001

    def fail_import(array, schema):
        if consume:
            imported.append(import_from_c(array, schema))
        message = "injected Arrow import failure"
        raise RuntimeError(message)

    function = shapes_arrow if operation == "shapes" else contours_arrow
    kwargs = {} if operation == "shapes" else {"thresholds": [0, 2]}
    with monkeypatch.context() as patch:
        patch.setattr(pa, "RecordBatch", SimpleNamespace(_import_from_c=fail_import))
        with pytest.raises(RuntimeError, match="injected Arrow import failure"):
            function(source, **kwargs)
    if imported:
        # Moved buffers must survive the failed wrapper call and Rust cleanup.
        assert imported[0].column("value").to_pylist() == [
            1.0 if operation == "shapes" else 0.0
        ]
        assert shapely.from_wkb(imported[0].column("geometry")[0].as_py()).is_valid
    table = function(source, **kwargs)
    assert table.num_rows == 1
    assert shapely.from_wkb(table["geometry"][0].as_py()).is_valid


@pytest.mark.parametrize("first_zero", [0.0, -0.0])
def test_equal_thresholds_preserve_first_zero_sign(first_zero):
    result = contours(np.ones((2, 2)), [first_zero, -first_zero, 2.0])
    assert len(result) == 1
    assert np.signbit(result[0][1]) == np.signbit(first_zero)
