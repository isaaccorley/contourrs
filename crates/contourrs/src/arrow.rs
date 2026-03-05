//! Arrow export — builds a RecordBatch with WKB geometry + f64 value columns.
//!
//! Gated behind the `arrow` feature flag.

use arrow::array::{ArrayRef, BinaryArray, Float64Array, RecordBatch};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use geo_types::Polygon;
use std::sync::Arc;

/// Encode a geo_types::Polygon as ISO WKB (little-endian).
pub fn polygon_to_wkb(polygon: &Polygon<f64>) -> Vec<u8> {
    let mut buf = Vec::new();
    polygon_to_wkb_into(&mut buf, polygon);
    buf
}

/// Append a polygon's WKB encoding to an existing buffer.
///
/// Used for streaming WKB into a single contiguous buffer when building
/// Arrow arrays, avoiding per-polygon `Vec<u8>` allocations.
pub fn polygon_to_wkb_into(buf: &mut Vec<u8>, polygon: &Polygon<f64>) {
    let exterior = polygon.exterior();
    let interiors = polygon.interiors();
    let num_rings = 1 + interiors.len();

    // Reserve capacity for this polygon's WKB
    let total_points: usize = exterior.0.len() + interiors.iter().map(|r| r.0.len()).sum::<usize>();
    buf.reserve(9 + 4 + num_rings * 4 + total_points * 16);

    // Byte order: 1 = little-endian
    buf.push(1u8);
    // WKB type: Polygon = 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Number of rings
    buf.extend_from_slice(&(num_rings as u32).to_le_bytes());

    // Exterior ring
    write_ring(buf, &exterior.0);

    // Interior rings (holes)
    for hole in interiors {
        write_ring(buf, &hole.0);
    }
}

// Compile-time guarantee that Coord<f64> is exactly two contiguous f64s (no padding).
// Required for the unsafe reinterpret cast in write_ring on little-endian targets.
const _: () = assert!(std::mem::size_of::<geo_types::Coord<f64>>() == 16);

#[inline]
fn write_ring(buf: &mut Vec<u8>, coords: &[geo_types::Coord<f64>]) {
    buf.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    // On little-endian: Coord<f64> memory layout matches WKB LE format directly.
    // Reinterpret the entire coords slice as bytes in one shot.
    #[cfg(target_endian = "little")]
    {
        // SAFETY: Coord<f64> is two contiguous f64s with no padding (verified by
        // the compile-time assert above) on LE targets, matching WKB LE layout.
        let byte_len = std::mem::size_of_val(coords);
        let ptr = coords.as_ptr() as *const u8;
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
pub fn polygons_to_record_batch(
    polygons: &[(Polygon<f64>, f64)],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    // Stream all WKB into a single buffer with offset tracking
    let mut wkb_data: Vec<u8> = Vec::new();
    let mut offsets: Vec<i32> = Vec::with_capacity(polygons.len() + 1);
    offsets.push(0);

    let mut values: Vec<f64> = Vec::with_capacity(polygons.len());

    for (polygon, value) in polygons {
        polygon_to_wkb_into(&mut wkb_data, polygon);
        debug_assert!(
            wkb_data.len() <= i32::MAX as usize,
            "WKB buffer exceeds i32::MAX ({} bytes)",
            wkb_data.len()
        );
        offsets.push(wkb_data.len() as i32);
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
        .collect(),
    );

    // GeoParquet schema-level metadata for compatibility
    let geo_meta = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Polygon"]}}}"#;

    let schema = Schema::new(vec![
        geometry_field,
        Field::new("value", DataType::Float64, false),
    ])
    .with_metadata(
        [("geo".to_string(), geo_meta.to_string())]
            .into_iter()
            .collect(),
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
