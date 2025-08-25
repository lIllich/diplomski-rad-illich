import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import warnings
import csv
from datetime import datetime
import time

def generate_audio_speecht5(
    text,
    output_file_path,
    tts_model_id="derek-thomas/speecht5_finetuned_voxpopuli_hr",
    vocoder_model_id="microsoft/speecht5_hifigan",
    models_dir="./models/",
    gpu_id=0
):
    
    warnings.filterwarnings("ignore", message="`huggingface_hub` cache-system uses symlinks by default")
    warnings.filterwarnings("ignore", message="Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed")
    warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")
    warnings.filterwarnings("ignore", message="`cache.key_cache[idx]` is deprecated")
    warnings.filterwarnings("ignore", message="`cache.value_cache[idx]` is deprecated")

    os.makedirs(models_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        if not hasattr(generate_audio_speecht5, '_device_set'):
            print(f"Korištenje GPU-a: {gpu_id}")
            generate_audio_speecht5._device_set = True
    else:
        device = "cpu"
        if not hasattr(generate_audio_speecht5, '_device_set'):
            print("Korištenje uređaja: CPU (CUDA nije dostupna ili nije odabrana)")
            generate_audio_speecht5._device_set = True
        
    # Učitavanje modela i vocodera
    if not hasattr(generate_audio_speecht5, 'processor'):
        print(f"Preuzimanje i učitavanje SpeechT5 modela '{tts_model_id}' u {models_dir}...")
        try:
            generate_audio_speecht5.processor = SpeechT5Processor.from_pretrained(tts_model_id, cache_dir=models_dir)
            generate_audio_speecht5.model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_id, cache_dir=models_dir)
            print("SpeechT5 model uspješno učitan.")
        except Exception as e:
            print(f"Greška pri učitavanju SpeechT5 modela: {e}")
            print("Provjerite imate li ispravan 'tts_model_id' i internetsku vezu za prvo preuzimanje.")
            raise
        
        print(f"Preuzimanje i učitavanje vocoder modela '{vocoder_model_id}' u {models_dir}...")
        try:
            generate_audio_speecht5.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_id, cache_dir=models_dir)
            print("Vocoder model uspješno učitan.")
        except Exception as e:
            print(f"Greška pri učitavanju vocoder modela: {e}")
            print("Provjerite imate li ispravan 'vocoder_model_id' i internetsku vezu za prvo preuzimanje.")
            raise

        generate_audio_speecht5.model.to(device)
        generate_audio_speecht5.vocoder.to(device)

        print("Dohvaćanje speaker embeddinga...")
        try:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            generate_audio_speecht5.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
            print("Speaker embedding učitan.")
        except Exception as e:
            print(f"Greška pri dohvaćanju speaker embeddinga: {e}")
            print("Provjerite imate li instaliranu 'datasets' biblioteku (`pip install datasets`).")
            print("Također, provjerite internetsku vezu za preuzimanje dataseta.")
            raise
    
    processor = generate_audio_speecht5.processor
    model = generate_audio_speecht5.model
    vocoder = generate_audio_speecht5.vocoder
    speaker_embeddings = generate_audio_speecht5.speaker_embeddings

    inputs = processor(text=text, return_tensors="pt").to(device)

    print(f"Generiranje govora za ID '{os.path.basename(output_file_path).split('_')[0]}'...")
    
    start_time = time.time()
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    end_time = time.time()
    
    generation_duration = end_time - start_time
    print(f"Trajanje generiranja: {generation_duration:.2f} sekundi.")

    audio_data = speech.squeeze().cpu().numpy()
    
    max_val = torch.max(torch.abs(speech)).item()
    if max_val > 1e-6:
        audio_data = audio_data / max_val
    else:
        audio_data = audio_data * 0

    # Spremanje audio datoteke
    sf.write(output_file_path, audio_data, samplerate=vocoder.config.sampling_rate)
    print(f"Audio datoteka spremljena kao: {output_file_path}\n")

if __name__ == "__main__":
    csv_file_path = "sinteza-govora-popis.csv"
    output_base_dir = "./outputs"
    
    os.makedirs(output_base_dir, exist_ok=True)

    print("Inicijalizacija modela, vocodera i speaker embeddinga...")
    try:
        # Dummy poziv za inicijalizaciju i preuzimanje modela
        # Prosljeđujemo putanju za privremenu datoteku, ali ona se ne koristi, jer se ne sprema.
        generate_audio_speecht5(
            "dummy text for initialization", 
            os.path.join(output_base_dir, "temp_dummy_for_init.wav")
        )
        print("Inicijalizacija završena. Modeli i resursi su učitani u memoriju.\n")
    except Exception as e:
        print(f"Greška pri inicijalizaciji: {e}")
        print("Nije moguće nastaviti s obradom CSV datoteke.")
        exit()


    print(f"Čitanje tekstova iz '{csv_file_path}' i generiranje audio datoteka...")
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            header = next(reader)
            
            try:
                text_col_idx = header.index("tekst")
                id_col_idx = header.index("id")
            except ValueError as e:
                print(f"Greška: Kolona 'tekst' ili 'id' nije pronađena u zaglavlju CSV datoteke. {e}")
                exit()

            for row in reader:
                if len(row) > text_col_idx and len(row) > id_col_idx:
                    record_id = row[id_col_idx]
                    input_text = row[text_col_idx]
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    
                    output_filename = f"{record_id}_{timestamp}.wav"
                    output_file_path = os.path.join(output_base_dir, output_filename)
                    
                    generate_audio_speecht5(
                        input_text,
                        output_file_path=output_file_path,
                    )
                else:
                    print(f"Upozorenje: Redak je preskočen zbog nedovoljnog broja kolona: {row}")

    except FileNotFoundError:
        print(f"Greška: Datoteka '{csv_file_path}' nije pronađena. Provjerite putanju.")
    except Exception as e:
        print(f"Dogodila se neočekivana greška prilikom obrade CSV datoteke: {e}")

    print("Proces generiranja audio datoteka je završen.")