# Changelog

## [0.4.0](https://github.com/isaaccorley/contourrs/compare/v0.3.0...v0.4.0) (2026-03-04)


### Features

* upgrade pyo3 0.23→0.28 for Python 3.14 support ([#5](https://github.com/isaaccorley/contourrs/issues/5)) ([5b441e8](https://github.com/isaaccorley/contourrs/commit/5b441e847ea8440ff2d4dc84b3028b6e519e2ca1))

## [0.3.0](https://github.com/isaaccorley/contourrs/compare/v0.2.0...v0.3.0) (2026-03-04)


### Features

* upgrade pyo3 0.22→0.23 for Python 3.14 support ([#3](https://github.com/isaaccorley/contourrs/issues/3)) ([5ca8fda](https://github.com/isaaccorley/contourrs/commit/5ca8fda224d82853dec3d2317f382023ddebbe8c))

## [0.2.0](https://github.com/isaaccorley/contourrs/compare/v0.1.0...v0.2.0) (2026-03-04)


### Features

* add example notebook, numpy docstrings, notebook CI ([5f75b4c](https://github.com/isaaccorley/contourrs/commit/5f75b4ce3d9633d6ca12b061a8a33ac54e35369b))
* add marching squares isoband contouring (contours/contours_arrow) ([6fbdb17](https://github.com/isaaccorley/contourrs/commit/6fbdb172c43a2dea4f6480e2112a0bf3f3e767d7))
* contourrs — fast raster polygonization with Arrow export ([b9c5d73](https://github.com/isaaccorley/contourrs/commit/b9c5d73dff8dd86376212d32df2210f55b47b013))


### Bug Fixes

* ignore integration tests in CI, skip geopandas when unavailable ([4128064](https://github.com/isaaccorley/contourrs/commit/4128064345e981a7049f898ebb071ca90e125150))
* swap marching squares cases 13/14, add GeoArrow metadata, update notebook ([2e40cf9](https://github.com/isaaccorley/contourrs/commit/2e40cf92945ea6b8bbf163b18eda79f8f92cfe8a))


### Performance Improvements

* round-2 optimizations — rayon parallel marching squares, bbox indexing, bulk WKB writes ([03c2bab](https://github.com/isaaccorley/contourrs/commit/03c2babb75ba706aeaf9441caba3752b142821b4))
