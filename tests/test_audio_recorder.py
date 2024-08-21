import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import time
from audio_processer import AudioRecorder

class TestAudioRecorder(unittest.TestCase):
  @patch('audio_processer.audio_recorder.sd.InputStream')
  def test_start_recording(self, mock_inputstream):
    mock_inputstream.return_value.__enter__.return_value = MagicMock()

    recorder = AudioRecorder()
    recorder.start_recording()

    self.assertTrue(recorder.is_recording)
    self.assertFalse(recorder.is_paused)
    self.assertIsNotNone(recorder.audio_thread)
    self.assertEqual(recorder.recordings, [])

  @patch('audio_processer.audio_recorder.sd.InputStream')
  def test_stop_recording(self, mock_inputstream):
      mock_inputstream.return_value.__enter__.return_value = MagicMock()

      recorder = AudioRecorder()
      recorder.start_recording()
      time.sleep(1)
      mock_data = np.array([[0.1], [0.2], [0.3]])
      recorder.recordings.append(mock_data)
      result = recorder.stop_recording()

      self.assertFalse(recorder.is_recording)
      self.assertIsNotNone(result)
      self.assertTrue(np.array_equal(result, mock_data))

  def test_toggle_recording(self):
      recorder = AudioRecorder()
      recorder.start_recording()

      self.assertFalse(recorder.is_paused)
      recorder.toggle_recording()
      self.assertTrue(recorder.is_paused)
      recorder.toggle_recording()
      self.assertFalse(recorder.is_paused)

  @patch('audio_processer.audio_recorder.sd.InputStream')
  def test_callback(self, mock_inputstream):
      mock_inputstream.return_value.__enter__.return_value = MagicMock()

      recorder = AudioRecorder()
      recorder.start_recording()

      # Simulate the callback being triggered with mock data
      mock_data = np.array([[0.1], [0.2]])
      recorder._callback(mock_data, None, None, None)

      self.assertEqual(len(recorder.recordings), 1)
      self.assertTrue(np.array_equal(recorder.recordings[0], mock_data))

if __name__ == "__main__":
    unittest.main()
