## Cached example rasters

This directory contains a small cached DEM used by `examples/dem_contour.py`.

- `mt_rainier_dem_2048.tif` (2048 x 2048, float32, EPSG:4269)
  - Source: USGS 3DEP 1/3 arc-second DEM (`n47w122` tile)
  - Upstream URL:
    `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n47w122/USGS_13_n47w122.tif`
  - Window center: Mount Rainier summit (~46.8528, -121.7604)

To regenerate this file, crop a 2048 x 2048 window from the source tile with
`rasterio` and save as a tiled, deflate-compressed GeoTIFF.
