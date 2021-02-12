# Text-Recognition_using_OpenCV-OCR

OpenCV OCR text recognition is performed using Tesseract-v4 - a highly popular OCR engine - which uses deep learning model having relatively high accuracy and speed. Prior to text recognition, text detection is performed using OpenCV's EAST text detector - a robust model, capable of localizing text even when it’s blurred, reflective, or partially obscured. Then, each of the text ROIs is extracted and passed through Tesseract to complete the task.


EAST (Efficient and Accurate Scene Text) detector is capable to work perfectly with orientations on 720p with 13 FPS. However, Tesseract doesn't work up to that mark, particularly in the presence of noise, haze, complex fonts and orientations. To address this issue, check out the [deep learning based OCR](https://github.com/Sudarshana2000/Sudoku-solver/blob/master/sudokunet.py).


## Results

<div style="float:left">
<div style="float:left"><img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/IMG1.jpeg" />
<img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/output1.jpeg" />
</div>
<br /><br />

<div style="float:left">
<div style="float:left"><img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/IMG2.jpg" />
<img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/output2.jpg" />
</div>
<br /><br />

<div style="float:left">
<div style="float:left"><img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/IMG3.jpg" />
<img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/output3.jpg" />
</div>
<br /><br />

<div style="float:left">
<div style="float:left"><img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/IMG4.jpg" />
<img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/output4.jpg" />
</div>
<br /><br />

<div style="float:left">
<div style="float:left"><img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/IMG5.jpg" />
<img width="45%" src="https://github.com/Sudarshana2000/Text-Recognition_using_OpenCV-OCR/blob/master/images/output5.jpg" />
</div>
<br /><br />