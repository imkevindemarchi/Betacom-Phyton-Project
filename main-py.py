'''
DEMO RAG

cosa fare:
1. leggere il documento
2. dividere in chunks
3. trasforma i chunk in vettori
4. data una domanda, recupera i chunk simili
5. passare i chunk all'LLM, elaborazione della domanda (chunk simili + prompt)


'''

# IMPORT
from pathlib import Path
import numpy as np
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings

# SETTAGGIO IPERPARAMETRI
CARTELLA_PROGETTO = Path(__file__).parent
PERCORSO_PDF = CARTELLA_PROGETTO / "documenti" / "java_teoria.pdf"
CARTELLA_MODELLI = CARTELLA_PROGETTO / "modelli"
NOME_MODELLO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

NOME_MODELLO_LLM1 = "llama3.2:1b"
NOME_MODELLO_LLM2 = "llama3.2:3b"

DIMENSIONE_CHNK = 800
SOVRAPPOSIZIONE = 150
NUMERO_CHUNK_CONTESTO = 4
TEMPERATURA = 0.2


# -----------------------------------------------------
#   1. LEGGERE I DOCUMENTI
# -----------------------------------------------------
def leggi_pdf(percorso):
    lettore = PdfReader(str(percorso))
    pagine = []
    for pagina in lettore.pages:
        pagine.append(pagina.extract_text() or "")
    return "\n".join(pagine)


def leggi_documento(percorso):
    if not percorso.exists():
        raise FileNotFoundError("File non trovato", percorso)

    estensione = percorso.suffix.lower()

    if estensione == ".pdf":
        testo = leggi_pdf(percorso)
    elif estensione == ".txt":
        testo = percorso.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError("Estensione non valida")

    if not testo.strip():
        raise ValueError("Testo vuoto")

    return testo


# -----------------------------------------------------
#   2. DIVIDERE IN CHUNKS
# -----------------------------------------------------
def dividi_in_chunk(testo, dimensione, sovrapposizione):
    testo = " ".join(testo.split())
    chunk = []
    inizio = 0

    while inizio < len(testo):
        fine = inizio + dimensione
        chunk.append(testo[inizio:fine])
        inizio += dimensione - sovrapposizione

    return chunk


# -----------------------------------------------------
#   3. CREARE GLI EMBEDDING
# -----------------------------------------------------
def carica_modello_embedding():
    # scarica il modello
    print("Scaricando il modello ... ... ... ...")
    return SentenceTransformer(
        NOME_MODELLO_EMBEDDING,
        cache_folder=str(CARTELLA_MODELLI)
    )


def crea_embedding(modello_embedding, lista_testi):
    return modello_embedding.encode(
        lista_testi,
        normalize_embeddings=True
    )


# -----------------------------------------------------
#   4. RECUPERARE I CHUNK SIMILI
# -----------------------------------------------------
def cerca_chunk_migliori(domanda, modello_embedding, chunk, vettori_chunk, k):
    # restituisce i k-chunk più "simili" alla domanda con score [0...1]
    vettore_domanda = crea_embedding(modello_embedding, [domanda])[0]
    similarita = vettori_chunk @ vettore_domanda
    indici_migliori = np.argsort(similarita)[::1][:k]  # posizione dei k punteggi più alti

    return [(chunk[i], float(similarita[i])) for i in indici_migliori]


# -----------------------------------------------------
#   5. GENERARE LA RISPOSTA (INTEGRAZIONE CON OLLAMA)
# -----------------------------------------------------
def costruisci_messaggi(domanda, passaggi):
    contesto = "\n".join(
        f"[Passaggio {n}\n{testo}" for n, testo in enumerate(passaggi, start=1)
    )

    istruzioni = (
        "Sei un assistente che risponde in italiano a domande inerenti a dei documenti."
        "Usa ESCLUSIVAMENTE le informazioni presenti nei passaggi forniti"
        "Se la risposta non è presente nei messaggi, rispondi: "
        "\"Non ho trovato la risposta nel documento \""
        "Non inventare e non utilizzare conoscenze esterne"
    )

    return [
        {
            "role": "system",
            "content": istruzioni,
        },
        {
            "role": "user",
            "content": f"PASSAGGI:\n{contesto}\n\nDomanda: {domanda}",
        }
    ]


def genera_risposta(domanda, passaggi, modello):
    messaggi = costruisci_messaggi(domanda, passaggi)
    try:
        flusso = ollama.chat(
            model=modello,
            messages=messaggi,
            stream=True,  # riceve la risposta pezzo per pezzo
            options={
                "temperature": TEMPERATURA,
            }
        )

        for parte in flusso:
            print(parte["message"]["content"], end="", flush=True)
        print()

    except Exception as e:
        print("Errore: ", e)


# ------------------
#       MAIN
# -----------------
def main():
    print(f"Leggo il documento: {PERCORSO_PDF.name}")
    testo = leggi_documento(PERCORSO_PDF)
    chunk = dividi_in_chunk(testo, DIMENSIONE_CHNK, SOVRAPPOSIZIONE)
    print("Documento pronto, diviso in ", len(chunk))

    modello_embedding = carica_modello_embedding()
    vettori_chunk = crea_embedding(modello_embedding, chunk)

    # ciclo di domande e risposte
    while True:
        domanda = input("\n Domanda >").strip()
        if domanda.lower() in ("esci", "exit", "quit", "q"):
            print("Ciao alla prossima")
            break

        if not domanda:
            continue

        pasaggi = cerca_chunk_migliori(
            domanda, modello_embedding, chunk, vettori_chunk, NUMERO_CHUNK_CONTESTO
        )

        print("\n RISPOSTA LLM1 >")
        genera_risposta(domanda, pasaggi, NOME_MODELLO_LLM1)
        print("\n RISPOSTA LLM2 >")
        genera_risposta(domanda, pasaggi, NOME_MODELLO_LLM2)


if __name__ == "__main__":
    main()
