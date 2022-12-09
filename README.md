# Blind and Visually Impaired Form Fill-out

The purpose of this code is to enable a blind person to be able to fill out paperwork independently.</br>

## Requirements:

- `pip install pyttsx3`
- `pip install mediapipe`
- `pip install pytesseract`
- `pip install opencv-python`
- `pip install numpy`
- `pip install imutils`
- `pip install skimage`

## To Run:

1. Print a piece of paper with the word "Sign" on it.
2. Enable sound on your computer and turn up the Volume
3. Create a backing for the paper and a way to hold the paper so as to not obstruct its edges.
4. Run Python 3 file: `Scan_Document.py`
5. Move the paper around until the program finds the word "Sign" and has identified the top left side of the paper. The program will auditorily alert you.
6. After the program finds where the user must sign, move your hand into the image and the audio should direct you to the stop on the paper where the signature should start.

## Sources:

- https://pointotech.blogspot.com/2019/03/cam-scanner-using-python.html
- https://google.github.io/mediapipe/getting_started/python.html
