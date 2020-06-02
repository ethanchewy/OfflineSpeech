# ALL CREDIT GOES TO Mozilla's Deepspeech: https://github.com/mozilla/DeepSpeech-examples/blob/r0.7/mic_vad_streaming/mic_vad_streaming.py
# This is simply a modification for our own research
# Our modifications allow for a streamline process for downloading all audio collected and storing all transcripts in an organized fashion in SQLITE


# To run continually, simply run
#       python script.py
# To continually run in the background, type:
#           TBD

# To read from current directory:
#           TBD
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal

# my libs
import sqlite3
import sounddevice as sd
import soundfile as sf
import threading
from pathlib import Path

# Used for reference in the Transcript databse
CURRENT_AUDIO_ID = 1

def collectAudio():
    # Start Timer
    timer_start = time.time()
    # Initially start recording
    # https://stackoverflow.com/questions/39474111/recording-audio-for-specific-amount-of-time-with-pyaudio
    samplerate = 44100  # Hertz
    duration = 10  # seconds

    # Temporarily just use a while true to get audio
    # might have to open another process for this
    while True:
        global CURRENT_AUDIO_ID
        n = datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")
        filename = os.path.join("data", n)
        Path(filename).touch()

        s = datetime.utcnow().timestamp()
        mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                channels=2, blocking=True)
        e = datetime.utcnow().timestamp()
        CURRENT_AUDIO_ID+=1

        sf.write(filename, mydata, samplerate)

        conn = sqlite3.connect("main.db")
        c = conn.cursor()
        conn.execute("INSERT INTO AUDIO (FILENAME, STARTTIME, ENDTIME, DURATION) VALUES (?, ?, ?, ?)", (n, s, e, duration*1000))
        conn.commit()
        conn.close()

def transcribe(ARGS):
    main(ARGS)


logging.basicConfig(level=20)

# SETTINGS FOR VOICE FILTERING
AGGRESSIVENESS = 3


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        # Add instance attribute for 10 second audio blocks
        # self.audio_blocks_queue = queue.Queue()
        # You can just record directly

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def main(ARGS):
    # Create database and tables have been created
    # > TABLE AUDIO=> keeps track of audio files
    # > TABLE transcripts => keeps track of transcripts
    #       For transcripts, the associated audio ids will be a text seperated by spaces since lists aren't supported in SQL
    # Time stored as integer
    # Duration is in milliseconds.

    # BY DEFAULT, WE USE MAIN.DB. CHANGE BELOW TO A NEW DB FILE IF NEED BE
    conn = sqlite3.connect("main.db")
    print("Opened main.db")
    c = conn.cursor()

    # TODO: consolidate code => too repetitive right now
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='AUDIO'").fetchone():
        print("Audio table does not exist.")
        print("Creating Audio Table...")
        conn.execute('''CREATE TABLE AUDIO
         (ID INTEGER PRIMARY KEY NOT NULL,
         FILENAME           TEXT    NOT NULL,
         STARTTIME            REAL     NOT NULL,
         ENDTIME        REAL     NOT NULL,
         DURATION       INT     NOT NULL);''')
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TRANSCRIPTS'").fetchone():
        print("Transcripts table does not exist.")
        print("Creating Transcripts Table...")
        conn.execute('''CREATE TABLE TRANSCRIPTS
         (ID INTEGER PRIMARY KEY NOT NULL,
         AUDIOIDS           TEXT    NOT NULL,
         STARTTIME            REAL     NOT NULL,
         ENDTIME        REAL     NOT NULL,
         TRANSCRIPT       TEXT,
         SPEAKER       TEXT);''')

    conn.commit()
    conn.close()
    #### SQL TABLE CREATION DONE

    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    print('Initializing model...')
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()

    #SQL Schema Stuff
    new_transcript = True
    transcript_start_time = 0
    transcript_end_time = 0
    transcript_text = ""
    #TBD
    # transcript_audio_ids = None

    for frame in frames:
        if frame is not None:
            if new_transcript:
                new_transcript = False
                transcript_start_time = datetime.utcnow().timestamp()
                print("Started Transcript", transcript_start_time)

            if spinner:
                spinner.start()
            # else:
                # print("test2")

            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
        else:
            new_transcript = True
            transcript_end_time = datetime.utcnow().timestamp()
            print("End Transcript", transcript_end_time)

            if spinner: spinner.stop()
            logging.debug("end utterence")
            # if ARGS.savewav:

                # vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                # wav_data = bytearray()
            # By default, write to file
            # print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"), wav_data)
            # vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
            # wav_data = bytearray()
            text = stream_context.finishStream()
            transcript_text = text
            # Generate audio ids
            audio_ids = ""
            counter = 0
            notDone = True
            difference = int(transcript_end_time - transcript_start_time)
            # DEFAULT IS 10 SECONDS
            while notDone:
                if difference <= 10:
                    notDone = False
                else:
                    difference -= 10
                    counter += 1
                if audio_ids == "":
                    audio_ids += str(CURRENT_AUDIO_ID)
                else:
                    audio_ids = audio_ids + " " + str(CURRENT_AUDIO_ID+counter)


            # TODO:
            # add time stamp to file
            conn = sqlite3.connect("main.db")
            c = conn.cursor()
            conn.execute("INSERT INTO TRANSCRIPTS (AUDIOIDS, STARTTIME, ENDTIME, TRANSCRIPT, SPEAKER) VALUES (?, ?, ?, ?, ?)", (audio_ids, transcript_start_time, transcript_end_time, transcript_text, "tbd"))
            conn.commit()
            conn.close()
            #####

            # For now also add wav files directly related to the text generated
            # We can also figure out a way to write to files every 10 seconds
            print("Recognized: %s" % text)
            stream_context = model.createStream()

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    # main(ARGS)
    # https://stackoverflow.com/questions/38254172/infinite-while-true-loop-in-the-background-python
    b = threading.Thread(name='background', target=collectAudio)
    f = threading.Thread(name='foreground', target=transcribe, args=(ARGS, ))
    b.start()
    f.start()
