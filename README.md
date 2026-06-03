# Python tools for managing Music datasets for OMR.

This repository has a set of tools to manage, maintain and expand existing datasets for the purpose of training OMR models. The toolkit consists of the following packages:
- kern: parsing humdrum kern files,
- midi: parse, create and edit midi files,
- imslp: grab sheet music from the IMSLP site,
- editor: create and edit annotatipons layered on top of existing sheet music, eg page structure informations such as staves.

Dataset sources:
- PDMX: https://zenodo.org/records/14648209
    Requires downloading multiple files, at least PDMX.csv, mxl.tar.gz and metadata.tar.gz
    
Examples:

To extract a reasonable subset from PDMX, you can use something like this:

```bash
# Select scores that have all pages rendering less than 16 staves.
pdmx query -o Staff16.csv 'index==index' --score 'pages.*.staff_count < 16' --valid
# Displays subset general statistics.
pdmx --csv subset.csv stats
# Train the staffer on that dataset.
staffer --log-file logs/staffer.log train -e 12 --use-sampler model_name
```

For the noter model, the same process applies:

```bash
# Select all scores that have systems of only one staff:
pdmx query -o System2.csv 'index==index' --score 'pages.0.systems.0.staff_count <= 2' --valid --bar-match
```

TODO List:
- Simplify the network output so staff becomes two coordinates only (top, bottom) derive other coordinates from the system
- import editor from projects/Staffer or projects/OMR
Pending fixes:
- The tokenizer should check the length of the first bars against the metric and decide based on that where the number 1 falls.
- In mxl/14/10/QmWAGXyEP8SJRRRPSy5jpFvX9MRGPqPUuHkUay19hAy8wM.mxl the rendering is missing the first few bars and is therefore out of sync.
- In /mxl/3/6/Qmd7UQFcdQg8fjqqCkJPHkc2N4PqQEkx6vh5sxqchozJu8.mxl the bar count mismatches likely because they are some invisible bars at the beginning of the svg file that the LayoutExtractor counts (it shouldn't).
- `StafferDataset`: ~93 samples have `box.bottom > page.image_height` (corrupted layout). Currently `max_sys_bottom` is clamped to 1.0, but corrupted box coords still flow into training ground truth via `__getitem__` (producing normalised coords > 1.0). These samples should be dropped entirely, not clamped.

