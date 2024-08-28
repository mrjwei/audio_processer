import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from pyaudiocook import AudioTranscriber

class TestAudioTranscriber(unittest.TestCase):
  def test_initialization(self):
    transcriber = AudioTranscriber(Path('audio1.wav'), mode='local')
    self.assertEqual(transcriber.mode, 'local')
    self.assertEqual(transcriber.filepaths, (Path('audio1.wav'),))
    self.assertEqual(transcriber.texts, [])

    with self.assertRaises(Exception):
        AudioTranscriber(Path('audio1.wav'), mode='invalid_mode')

  @patch('pyaudiocook.audio_transcriber.whisper')
  def test_transcribe_local(self, mock_whisper):
      mock_model = MagicMock()
      mock_model.transcribe.return_value = {'text': 'Test transcript'}
      mock_whisper.load_model.return_value = mock_model

      transcriber = AudioTranscriber(Path('audio1.wav'), mode='local')
      transcriber.transcribe()

      mock_whisper.load_model.assert_called_once_with('tiny.en')
      mock_model.transcribe.assert_called_once_with('audio1.wav')
      self.assertEqual(transcriber.texts, ['Test transcript'])

  @patch('pyaudiocook.audio_transcriber.OpenAI')
  @patch('builtins.open', new_callable=mock_open)
  def test_transcribe_remote(self, mock_open_file, mock_openai):
      # Mock OpenAI client and its response
      mock_client = MagicMock()
      mock_response = MagicMock()
      mock_response.text = 'Test transcript'
      mock_client.audio.transcriptions.create.return_value = mock_response
      mock_openai.return_value = mock_client

      # Initialize the AudioTranscriber in remote mode
      transcriber = AudioTranscriber(Path('audio1.wav'), mode='remote')

      # Call the transcribe method
      transcriber.transcribe()

      # Check if OpenAI client and file opening were called correctly
      mock_openai.assert_called_once_with()
      mock_client.audio.transcriptions.create.assert_called_once_with(
          model='whisper-1',
          file=mock_open_file.return_value,
          language='en'
      )
      self.assertEqual(transcriber.texts, ['Test transcript'])


if __name__ == "__main__":
    unittest.main()
