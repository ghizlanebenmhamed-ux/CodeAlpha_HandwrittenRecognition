import unittest
from unittest.mock import Mock
import numpy as np
from PIL import Image,ImageDraw
from model import Recognizer,prepare_drawing,ROOT
from app import DigitApp

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.recognizer=Recognizer();cls.samples=np.load(ROOT/'data'/'test_samples.npz')
    def test_blank(self):
        with self.assertRaises(ValueError):prepare_drawing(Image.new('L',(280,280)))
    def test_preprocessing(self):
        image=Image.new('L',(280,280));ImageDraw.Draw(image).line((100,20,100,220),fill=255,width=16)
        prepared=prepare_drawing(image);self.assertEqual(prepared.shape,(28,28));self.assertGreater(prepared.max(),0)
    def test_invalid_pixels(self):
        for pixels in [np.zeros((28,28)),np.ones((12,12)),np.full((28,28),np.nan),np.full((28,28),300)]:
            with self.assertRaises(ValueError):self.recognizer.predict(pixels)
    def test_sample_outputs(self):
        correct=0
        for pixels,label in zip(self.samples['images'],self.samples['labels']):
            digit,scores=self.recognizer.predict(pixels)
            self.assertTrue(0<=digit<=9);self.assertAlmostEqual(float(scores.sum()),1,places=5);correct+=digit==label
        self.assertGreater(correct/200,.9)
    def app(self):
        app=DigitApp.__new__(DigitApp);app.recognizer=self.recognizer;app.samples=self.samples;app.sample_index=0;app.result=Mock();app.image=Image.new('L',(280,280));app.canvas=Mock();return app
    def test_predict_handler(self):
        app=self.app();app.predict();self.assertIn('Predicted digit:',app.result.set.call_args.args[0]);self.assertIn('Recorded label:',app.result.set.call_args.args[0])
    def test_enter(self):
        app=self.app();app.predict(Mock());self.assertIn('Predicted digit:',app.result.set.call_args.args[0])
    def test_blank_handler(self):
        app=self.app();app.sample_index=None;app.predict();self.assertIn('Draw a digit',app.result.set.call_args.args[0])
    def test_clear(self):
        app=self.app();app.clear();self.assertIsNone(app.sample_index);self.assertIsNone(app.image.getbbox())

if __name__=='__main__':unittest.main(verbosity=2)
