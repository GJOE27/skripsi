import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pydub
from pydub import AudioSegment
from threading import Thread
import time  # Ditambahkan untuk simulasi

def getMFCC(audio_data, sample_rate):
  # A 1024-point STFT with frames of 64 ms and 75% overlap.
  stfts = tf.signal.stft(audio_data, frame_length=1024, frame_step=256,
                        fft_length=1024)
  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]
  return mfccs

class VowelClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("SISTEM KLASIFIKASI TIPE VOKAL")
        master.geometry("650x420")
        master.configure(bg="#4a4a4a")

        self.page1 = Page1(master, self)
        self.page2 = Page2(master, self)

        self.show_page1("Short Time Fourier Transform")

    
    def show_page1(self, option):
        # self.page1.hide()
        self.page2.hide()
        self.page1.show(option)

    def show_page2(self, prediction_result):
        # self.page1.hide()
        self.page1.hide()
        self.page2.show(prediction_result)


class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()
        self.frame.configure(bg="#4a4a4a")

        self.label = tk.Label(self.frame, text="Rekam suara anda", font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        self.record_button = tk.Button(self.frame, text="Mulai Rekam", font=("Helvetica", 28), bg="white", fg="black", command=self.record_audio)
        # self.back_button = tk.Button(self.frame, text="Back", font=("Helvetica", 25), bg="white", fg="black", command=app.show_page1)

        self.label.pack(pady=20)
        self.record_button.pack(pady=80)
        # self.back_button.pack(pady=40)

    def show(self, option):
        self.frame.pack()

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):
        
        for i in range(3, 0, -1):
            self.label.config(text=f"Perekaman dimulai dalam {i}")
            # self.master.update()
            time.sleep(1)

        self.label.config(text="Perekaman suara...")
        # self.master.update()

        recording_thread = Thread(target=self.simulate_recording)
        recording_thread.start()

    def simulate_recording(self):
        try:
            duration = 3
            file_name = "STFT.wav"

            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=44100,
                            input=True,
                            frames_per_buffer=1024)

            frames = []

            for i in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))

            self.label.config(text="Rekaman suara anda berhasil")
            # self.master.update()

            model = load_model('./skripsi/VocalClassifierAugmented.h5')
            audio_data, sample_rate = librosa.load('./skripsi/dataset_test/Alto_855.wav')
            mfccs = getMFCC(audio_data, sample_rate)

            X = np.array(mfccs)
            X = X.reshape(1, 255, 13, 1)


            y_prob = model.predict(X)
            prediction_result = y_prob.argmax(axis=1)

            if prediction_result == 0:
                prediction_result = "Alto"
            elif prediction_result == 1:
                prediction_result = "Bass"
            elif prediction_result == 2:
                prediction_result = "Sopran"
            elif prediction_result == 3:
                prediction_result = "Tenor"

            self.app.show_page2(prediction_result)


        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat merekam audio: {str(e)}")

class Page2:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()

        self.label = tk.Label(self.frame, text="Prediksi tipe vokal anda adalah:", font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        self.prediction_label = tk.Label(self.frame, text="")

        self.quit_button = tk.Button(self.frame, text="Back", font=("Helvetica", 25), bg="white", fg="black", command=lambda: app.show_page1("Short Time Fourier Transform"))

        self.label.pack(pady=20)
        self.prediction_label.pack(pady=80)
        self.quit_button.pack(pady=40)

    def show(self, prediction_result):
        self.prediction_label.config(text=prediction_result, font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        self.frame.pack()
        self.frame.configure(bg="#4a4a4a")

    def hide(self):
        self.frame.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = VowelClassificationApp(root)
    root.mainloop()
