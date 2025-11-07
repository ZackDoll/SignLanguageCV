# ASL Citizen Integration Guide

## Files
- **test.csv**: CSV File of 32,941 test samples
- **train.csv**: CSV File of 40,154 training samples
- **val.csv**: CSV File of 10,304 validation samples
- **videos Folder**: folder containing all videos in the training 
testing and validation data pieces

## Columns for CSV
 - Gross: Classification of sign
 - Participant ID: ID of the person signing
 - Video file: Name of the file in videos folder that matches to that data
 - ASL-VEX Code: ASL-VEX encoding of the file number

# MS-ASL Dataset Integration Guide

This guide explains how to use my sign language recognition system with the MS-ASL dataset.

UPDATED MODEL USES DIFFERENT DATA SET, SCROLL TO THAT IF USING UPDATED VERSION
## Download from this link:
https://www.microsoft.com/en-us/download/details.aspx?id=100121
## Dataset Structure

### Files
- **MSASL_train.json**: Array of 16,054 training samples
- **MSASL_test.json**: Array of 4,172 test samples  
- **MSASL_val.json**: Array of 5,287 validation samples
- **MSASL_classes.json**: Array of 1,000 class names: `["ticket", "nice", "teacher", ...]`
- **MSASL_synonym.json**: Array of synonym groups: `[["ticket", "give ticket"], ["get", "receive"], ...]`

### Sample Format
Each sample is a dictionary:
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "start_time": 0.0,
  "end_time": 1.969,
  "label": 805,
  "text": "beer",
  "box": [0.047, 0.290, 1.0, 0.823],
  "width": 640.0,
  "height": 360.0,
  "fps": 29.97,
  "signer_id": 20,
  ...
}

