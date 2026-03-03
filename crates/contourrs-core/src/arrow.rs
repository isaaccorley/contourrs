//! Arrow export — builds a RecordBatch with WKB geometry + f64 value columns.
//!
//! Gated behind the `arrow` feature flag.

use arrow::array::{ArrayRef, BinaryArray, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use geo_types::Polygon;
use std::sync::Arc;

/// Encode a geo_types::Polygon as ISO WKB (little-endian).
pub fn polygon_to_wkb(polygon: &Polygon<f64>) -> Vec<u8> {
    let exterior = polygon.exterior();
    let interiors = polygon.interiors();
    let num_rings = 1 + interiors.len();

    // Estimate capacity: header(9) + num_rings(4) + rings
    let total_points: usize = exterior.0.len() + interiors.iter().map(|r| r.0.len()).sum::<usize>();
    let mut buf = Vec::with_capacity(9 + 4 + num_rings * 4 + total_points * 16);

    // Byte order: 1 = little-endian
    buf.push(1u8);
    // WKB type: Polygon = 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Number of rings
    buf.extend_from_slice(&(num_rings as u32).to_le_bytes());

    // Exterior ring
    write_ring(&mut buf, &exterior.0);

    // Interior rings (holes)
    for hole in interiors {
        write_ring(&mut buf, &hole.0);
    }

    buf
}

fn write_ring(buf: &mut Vec<u8>, coords: &[geo_types::Coord<f64>]) {
    buf.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for c in coords {
        buf.extend_from_slice(&c.x.to_le_bytes());
        buf.extend_from_slice(&c.y.to_le_bytes());
    }
}

/// Build an Arrow RecordBatch from polygonize results.
///
/// Columns:
/// - `geometry`: Binary (WKB-encoded polygons)
/// - `value`: Float64
///
/// Includes GeoParquet-compatible schema metadata.
pub fn polygons_to_record_batch(
    polygons: &[(Polygon<f64>, f64)],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    // Encode all polygons to WKB
    let wkb_buffers: Vec<Vec<u8>> = polygons.iter().map(|(p, _)| polygon_to_wkb(p)).collect();

    let wkb_refs: Vec<&[u8]> = wkb_buffers.iter().map(|b| b.as_slice()).collect();
    let geometry_array = BinaryArray::from_vec(wkb_refs);

    // Values
    let values: Vec<f64> = polygons.iter().map(|(_, v)| *v).collect();
    let value_array = Float64Array::from(values);

    // Schema with GeoParquet metadata
    let geo_meta = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Polygon"]}}}"#;

    // GeoArrow extension type on geometry field (for GeoPandas from_arrow)
    let geometry_field = Field::new("geometry", DataType::Binary, false).with_metadata(
        vec![
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            ("ARROW:extension:metadata".to_string(), "{}".to_string()),
        ]
        .into_iter()
        .collect(),
    );

    let schema = Schema::new(vec![
        geometry_field,
        Field::new("value", DataType::Float64, false),
    ])
    .with_metadata(
        vec![("geo".to_string(), geo_meta.to_string())]
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
    }
}
