# Diplomski rad - Evaluacija jezičnih modela za potrebe audiorehabilitacije

Ovaj repozitorij sadrži sve relevantne materijale, skripte i rezultate povezane s diplomskim radom na temu **Evaluacija jezičnih modela za potrebe audiorehabilitacije**. Projekt uključuje generiranje jezičnog materijala pomoću velikih jezičnih modela (LLM) te sintezu govora (TTS) na hrvatskom jeziku, kao i evaluaciju dobivenih rezultata.

---

## Struktura repozitorija

- `izlazi/text-generation/`  
  Sadržaj: Datoteke generirane velikim jezičnim modelom (LLM), uključujući generirane riječi i rečenice uz fonetske specifikacije.

- `izlazi/text-to-speech/`  
  Sadržaj: Audio datoteke generirane modelom za sintezu govora (TTS), koji sintetizira glasovni zapis na temelju tekstualnih ulaza.

- `rezultati/`  
  Sadržaj: Rezultati evaluacije u različitim formatima:  
  - `rezultati.xls` — glavna Excel datoteka sa statističkim analizama  
  - `rezultati_klasifikacija.csv` — klasifikacijski podaci za generirane materijale  
  - `rezultati_testovi.csv` — podaci za statističke testove  
  - `sinteza-govora-popis.csv` — popis generiranih audio zapisa sa statusom njihove kvalitete

- `skripte/`  
  Sadržaj: Skripte korištene u istraživanju i obradi podataka:  
  - `cuda-test.py` — provjera dostupnosti i konfiguracije CUDA okruženja  
  - `DeepSeek-R1-Distill-Qwen-32B_v1.py` — skripta za pokretanje i izvođenje zadataka generiranja teksta na LLM-u  
  - `speecht5_finetuned_voxpopuli_hr_v1.py` — skripta za pokretanje sinteze govora (TTS) na hrvatskom jeziku

- `ulazi/`  
  Sadržaj: Ulazne datoteke koje sadrže zadatke za LLM model:  
  - `input_recenice.json` — zadaci za generiranje rečenica  
  - `input_rijeci.json` — zadaci za generiranje listi riječi

---

## Kako koristiti

1. **Generiranje teksta (LLM):**  
   Pokrenite skriptu `DeepSeek-R1-Distill-Qwen-32B_v1.py` s odgovarajućim ulaznim JSON datotekama (nalaze se u `ulazi/`). Generirani tekst bit će spremljen u direktorij `izlazi/text-generation/`.

2. **Sinesteza govora (TTS):**  
   Na izlazne tekstualne datoteke primijenite `speecht5_finetuned_voxpopuli_hr_v1.py` za generiranje audio snimki u `izlazi/text-to-speech/`.

3. **Evaluacija i analiza:**  
   Upotrijebite dostupne Excel i CSV datoteke u `rezultati/` za kvantitativnu i kvalitativnu procjenu performansi modela.

4. **Provjera računalnog okruženja:**  
   Skripta `cuda-test.py` može pomoći pri provjeri ispravne instalacije CUDA podrške za ubrzavanje izvođenja modela korištenjem GPU-a.

---

## Reference

Ovaj repozitorij prati eksperimentalni rad i analize opisane u diplomskom radu Luka Illicha pod nazivom _Evaluacija jezičnih modela za potrebe audiorehabilitacije_ (Sveučilište u Rijeci, Tehnički fakultet, 2025).

---

## Kontakt

Za dodatna pitanja ili prijedloge, molimo kontaktirajte autora rada.

---

*Napomena:* Ovaj repozitorij zahtijeva Python okruženje s instaliranim paketima `transformers`, `torch`, `librosa` i ostalim navedenim u diplomskom radu kako bi se moglo reproducirati generiranje i sinteza govora.
