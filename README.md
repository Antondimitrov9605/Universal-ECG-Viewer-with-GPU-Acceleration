# Universal ECG Viewer with GPU Acceleration

Universal ECG Viewer is a desktop application for visualization and analysis of electrocardiographic data from binary files. The program supports configurable number of leads (3, 5, 12 or more), automatic file format detection, and options for loading selected time segments from long-term recordings. 

It includes digital signal filtering (bandpass and notch filters), automatic heart rate calculation, data navigation with variable time window, scaling, and CSV export functionality. The application offers optional GPU acceleration via CUDA/CuPy for faster processing of large files, with automatic fallback to CPU mode when suitable hardware is unavailable.

## Features

- Support for multiple lead configurations (3, 5, 12+ leads)
- Automatic file format detection
- Selective loading of time segments
- Digital signal filtering (bandpass and notch)
- Automatic heart rate calculation
- Variable time window navigation
- Signal scaling and auto-scale
- CSV export
- Optional GPU acceleration (CUDA/CuPy)

## License

MIT License

Copyright (c) 2025 [A.Dimitrov]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Medical Disclaimer

This software is provided for research and educational purposes only. It is NOT intended for clinical diagnosis, treatment, or any medical decision-making. The authors and contributors assume no liability for any medical consequences arising from the use of this software. Always consult qualified healthcare professionals for medical advice and decisions.
