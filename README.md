# Learned Image Compression with Hierarchical Progressive Context Modeling

## Introduction

This repository is the unofficial PyTorch implementation of the paper *"Learned Image Compression with Hierarchical Progressive Context Modeling"* with `compressai` package.

## Environment

- Python 3.12.8
- Torch 2.4.0
- Compressai 1.2.6

## Comparison and Issue

Test on Base model.

**Notice**: Still facing problem in `z` bit rate counts, and the official test script crashes in MS-SSIM calculation.

| \lambda | y_bpp(our) | y_bpp(official) | PSNR(our) | PSNR(official)|
|--------------------------------------------------------------------|
|18|0.122986|0.121093|29.20|29.20|
|35|0.179546|0.177663|30.55|30.55|
|67|0.280181|0.278256|32.20|32.20|
|130|0.423536|0.421610|34.00|34.00|
|250|0.608096|0.606170|35.80|35.80| 
|483|0.856890|0.854938|37.54|37.54|   


## Usage

Download checkpoint from the official repository and run the `Rename.py` scripts

```
python Rename.py --origin_path /path/to/checkpoints --output_path /path/to/renamed/checkpoints
```



## Related Link

- [Official Repository](https://github.com/lyq133/LIC-HPCM)
