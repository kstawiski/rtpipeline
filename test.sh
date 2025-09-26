#!/bin/bash
# Working rtpipeline command with NumPy compatibility fixes
# This command now works with TotalSegmentator thanks to the numpy compatibility shim
rtpipeline --dicom-root Example_data --outdir ./Data_Organized --logs ./Logs -v --workers 4