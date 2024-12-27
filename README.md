# DashcamCrop

A tool for automatically cropping dashcam footage by tracking corner points.

## Demo
![DashcamCrop Demo](assets/demo.gif)

## Features
- Automatic corner point tracking using template matching
- Real-time corner point visualization
- Adjustable template size for better tracking
- Perspective correction
- Motion smoothing with low-pass filter

## Requirements
- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jujuspace/DashcamCrop.git
cd DashcamCrop
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py sample.mp4
```


### Advanced Options

```bash
python main.py --input video.mp4 --output cropped.mp4 --template-size 50 --smooth-factor 0.8
```


### Parameters
- `--input`: Input video file path
- `--output`: Output video file path
- `--template-size`: Size of template for corner tracking (default: 40)
- `--smooth-factor`: Smoothing factor for motion (0-1, default: 0.7)

## How It Works
1. The program first detects corner points in the initial frame
2. These points are tracked through subsequent frames using template matching
3. A perspective transform is applied to crop and correct the video
4. Motion is smoothed using a low-pass filter

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)