//! Arrow export — builds a RecordBatch with WKB geometry + f64 value columns.
//!
//! Gated behind the `arrow` feature flag.

use arrow::array::{ArrayRef, BinaryArray, Float64Array, RecordBatch};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use geo_types::Polygon;
use std::sync::Arc;

/// Append a polygon's WKB encoding to an existing buffer.
///
/// Used for streaming WKB into a single contiguous buffer when building
/// Arrow arrays, avoiding per-polygon `Vec<u8>` allocations.
///
/// # Panics
/// Panics if a ring or point count exceeds WKB's `u32` limit, or the encoded
/// size overflows `usize`. Use [`try_polygon_to_wkb_into`] for fallible encoding.
pub fn polygon_to_wkb_into(buf: &mut Vec<u8>, polygon: &Polygon<f64>) {
    try_polygon_to_wkb_into(buf, polygon).unwrap_or_else(|error| panic!("{error}"));
}

/// Append WKB after validating its counts and size, leaving `buf` intact on error.
///
/// # Errors
/// Returns an error if ring or point counts exceed `u32`, or the encoded size
/// or resulting buffer length overflows `usize`.
pub fn try_polygon_to_wkb_into(
    buf: &mut Vec<u8>,
    polygon: &Polygon<f64>,
) -> Result<(), ArrowError> {
    let size = wkb_size(polygon)?;
    buf.len()
        .checked_add(size)
        .ok_or_else(|| ArrowError::ComputeError("WKB buffer size overflows usize".into()))?;
    write_polygon(buf, polygon, size);
    Ok(())
}

fn wkb_size(polygon: &Polygon<f64>) -> Result<usize, ArrowError> {
    polygon
        .interiors()
        .len()
        .checked_add(1)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| ArrowError::ComputeError("WKB ring count exceeds u32::MAX".into()))?;
    let mut size = 9usize;
    for ring in std::iter::once(polygon.exterior()).chain(polygon.interiors()) {
        u32::try_from(ring.0.len())
            .map_err(|_| ArrowError::ComputeError("WKB point count exceeds u32::MAX".into()))?;
        size = ring
            .0
            .len()
            .checked_mul(16)
            .and_then(|bytes| bytes.checked_add(4))
            .and_then(|bytes| size.checked_add(bytes))
            .ok_or_else(|| ArrowError::ComputeError("WKB polygon size overflows usize".into()))?;
    }
    Ok(size)
}

fn next_offset(current: usize, additional: usize) -> Result<i32, ArrowError> {
    current
        .checked_add(additional)
        .and_then(|size| i32::try_from(size).ok())
        .ok_or_else(|| {
            ArrowError::ComputeError(format!(
                "WKB buffer exceeds i32::MAX ({current} + {additional} bytes)"
            ))
        })
}

fn write_polygon(buf: &mut Vec<u8>, polygon: &Polygon<f64>, size: usize) {
    let exterior = polygon.exterior();
    let interiors = polygon.interiors();
    let num_rings = 1 + interiors.len();

    // Reserve capacity for this polygon's WKB
    buf.reserve(size);

    // Byte order: 1 = little-endian
    buf.push(1u8);
    // WKB type: Polygon = 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Number of rings
    let num_rings = u32::try_from(num_rings).expect("wkb_size validated the ring count");
    buf.extend_from_slice(&num_rings.to_le_bytes());

    // Exterior ring
    write_ring(buf, &exterior.0);

    // Interior rings (holes)
    for hole in interiors {
        write_ring(buf, &hole.0);
    }
}

// Coord has Rust layout, so size alone does not guarantee x/y field order.
// Verify every assumption used by the little-endian bulk WKB copy.
const _: () = {
    assert!(
        std::mem::size_of::<geo_types::Coord<f64>>() == 16,
        "WKB bulk copy requires two packed f64 coordinates"
    );
    assert!(
        std::mem::offset_of!(geo_types::Coord<f64>, x) == 0,
        "WKB bulk copy requires x first"
    );
    assert!(
        std::mem::offset_of!(geo_types::Coord<f64>, y) == 8,
        "WKB bulk copy requires y immediately after x"
    );
};

#[inline]
fn write_ring(buf: &mut Vec<u8>, coords: &[geo_types::Coord<f64>]) {
    let count = u32::try_from(coords.len()).expect("wkb_size validated the point count");
    buf.extend_from_slice(&count.to_le_bytes());
    // On little-endian: Coord<f64> memory layout matches WKB LE format directly.
    // Reinterpret the entire coords slice as bytes in one shot.
    #[cfg(target_endian = "little")]
    {
        let byte_len = std::mem::size_of_val(coords);
        let ptr = coords.as_ptr().cast::<u8>();
        // SAFETY: The size and field-offset assertions above guarantee contiguous
        // initialized x/y f64s with no padding, matching WKB LE layout. The byte
        // slice borrows live coordinates and is copied before this borrow ends.
        buf.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, byte_len) });
    }
    #[cfg(not(target_endian = "little"))]
    {
        for c in coords {
            buf.extend_from_slice(&c.x.to_le_bytes());
            buf.extend_from_slice(&c.y.to_le_bytes());
        }
    }
}

/// Build an Arrow RecordBatch from polygonize results.
///
/// Columns:
/// - `geometry`: Binary (WKB-encoded polygons, GeoArrow `geoarrow.wkb` extension type)
/// - `value`: Float64
///
/// Streams all WKB into a single contiguous buffer (one allocation) instead
/// of N separate `Vec<u8>` per polygon.
///
/// # Errors
/// Returns an error when WKB counts or sizes cannot be represented, the geometry
/// buffer would exceed Arrow Binary's `i32::MAX` byte limit, or batch construction fails.
pub fn polygons_to_record_batch(
    polygons: &[(Polygon<f64>, f64)],
) -> Result<RecordBatch, ArrowError> {
    // Stream all WKB into a single buffer with offset tracking
    let mut wkb_data: Vec<u8> = Vec::new();
    let mut offsets: Vec<i32> = Vec::with_capacity(polygons.len() + 1);
    offsets.push(0);

    let mut values: Vec<f64> = Vec::with_capacity(polygons.len());

    for (polygon, value) in polygons {
        let size = wkb_size(polygon)?;
        let offset = next_offset(wkb_data.len(), size)?;
        write_polygon(&mut wkb_data, polygon, size);
        offsets.push(offset);
        values.push(*value);
    }

    // Build BinaryArray from raw buffers (zero-copy from our single allocation)
    let offset_buffer = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let data_buffer = Buffer::from(wkb_data);
    let geometry_array = BinaryArray::new(offset_buffer, data_buffer, None);

    let value_array = Float64Array::from(values);

    // GeoArrow extension type on geometry field
    let geometry_field = Field::new("geometry", DataType::Binary, false).with_metadata(
        [
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            ("ARROW:extension:metadata".to_string(), "{}".to_string()),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>(),
    );

    // No CRS can be inferred from an affine transform. Omitting this key
    // would incorrectly default to OGC:CRS84 in GeoParquet readers.
    let geo_meta = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Polygon"],"crs":null}}}"#;

    let schema = Schema::new(vec![
        geometry_field,
        Field::new("value", DataType::Float64, false),
    ])
    .with_metadata(
        [("geo".to_string(), geo_meta.to_string())]
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>(),
    );

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(geometry_array) as ArrayRef,
            Arc::new(value_array) as ArrayRef,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString};

    #[test]
    fn binary_offset_rejects_overflow_without_allocating() {
        let limit = i32::MAX as usize;
        assert_eq!(next_offset(limit - 1, 1).unwrap(), i32::MAX);
        assert!(next_offset(limit, 1).is_err());
        assert!(next_offset(usize::MAX, 1).is_err());
    }

    #[test]
    fn wkb_coordinates_match_portable_little_endian_encoding() {
        let ring: LineString<f64> = LineString(vec![
            Coord { x: -0.0, y: 1.5 },
            Coord {
                x: -3.25,
                y: 1.0e10,
            },
            Coord { x: 2.0, y: -8.5 },
            Coord { x: -0.0, y: 1.5 },
        ]);
        let polygon = Polygon::new(ring.clone(), vec![ring]);
        let mut expected = vec![1];
        expected.extend_from_slice(&3u32.to_le_bytes());
        expected.extend_from_slice(&2u32.to_le_bytes());
        for ring in std::iter::once(polygon.exterior()).chain(polygon.interiors()) {
            expected.extend_from_slice(&u32::try_from(ring.0.len()).unwrap().to_le_bytes());
            for coordinate in &ring.0 {
                expected.extend_from_slice(&coordinate.x.to_le_bytes());
                expected.extend_from_slice(&coordinate.y.to_le_bytes());
            }
        }
        let mut encoded = Vec::new();
        try_polygon_to_wkb_into(&mut encoded, &polygon).unwrap();
        assert_eq!(encoded, expected);
        assert_eq!(wkb_size(&polygon).unwrap(), encoded.len());
    }

    fn polygon_to_wkb(polygon: &Polygon<f64>) -> Vec<u8> {
        let mut buf = Vec::new();
        polygon_to_wkb_into(&mut buf, polygon);
        buf
    }

    #[test]
    fn test_wkb_roundtrip_simple() {
        let exterior = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let wkb = polygon_to_wkb(&polygon);

        // Verify header
        assert_eq!(wkb[0], 1); // little-endian
        assert_eq!(u32::from_le_bytes([wkb[1], wkb[2], wkb[3], wkb[4]]), 3); // Polygon
        assert_eq!(u32::from_le_bytes([wkb[5], wkb[6], wkb[7], wkb[8]]), 1); // 1 ring
        assert_eq!(u32::from_le_bytes([wkb[9], wkb[10], wkb[11], wkb[12]]), 5); // 5 points
    }

    #[test]
    fn test_record_batch() {
        let exterior = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let batch = polygons_to_record_batch(&[(polygon, 42.0)]).unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).name(), "geometry");
        assert_eq!(batch.schema().field(1).name(), "value");

        // Check GeoParquet metadata
        let schema = batch.schema();
        let meta = schema.metadata();
        assert!(meta.contains_key("geo"));

        // Check GeoArrow field metadata
        let geo_field = schema.field(0);
        let field_meta = geo_field.metadata();
        assert_eq!(
            field_meta.get("ARROW:extension:name").unwrap(),
            "geoarrow.wkb"
        );
    }

    #[test]
    fn test_streaming_wkb_matches_standalone() {
        let ext = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let hole = LineString(vec![
            Coord { x: 0.2, y: 0.2 },
            Coord { x: 0.8, y: 0.2 },
            Coord { x: 0.5, y: 0.8 },
            Coord { x: 0.2, y: 0.2 },
        ]);
        let polygon = Polygon::new(ext, vec![hole]);

        // Standalone
        let standalone = polygon_to_wkb(&polygon);

        // Streaming into existing buffer
        let mut buf = vec![0xDE, 0xAD]; // pre-existing data
        polygon_to_wkb_into(&mut buf, &polygon);

        assert_eq!(&buf[2..], &standalone[..]);
    }
}
