import os
import sounddevice as sd
import numpy as np
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from config import get_db_address, init_rag, whisper_model
from graph import workflow


def _handle_ingest(arg: str):
    from memory.vector_store import ingest_pdf, ingest_text
    path = arg.strip()
    if not os.path.exists(path):
        print(f"[ERRORE] File non trovato: {path}")
        return
    if path.lower().endswith(".pdf"):
        n = ingest_pdf(path)
        print(f"[RAG] PDF indicizzato: {n} chunk da '{os.path.basename(path)}'")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n = ingest_text(text, source=os.path.basename(path))
        print(f"[RAG] Testo indicizzato: {n} chunk da '{os.path.basename(path)}'")


def listen(
    samplerate: int = 16000,
    silence_threshold: float = 0.01,  # volume below this = silence
    silence_duration: float = 1.5,    # seconds of silence before stopping
    chunk_duration: float = 0.1,      # size of each recorded chunk in seconds
) -> np.ndarray:
    chunk_size = int(samplerate * chunk_duration)
    silence_chunks_needed = int(silence_duration / chunk_duration)

    print("🎤 In ascolto... (parla quando vuoi)")
    frames = []
    silent_chunks = 0
    speech_started = False

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            volume = np.abs(chunk).mean()

            if volume > silence_threshold:
                if not speech_started:
                    print("🔴 Voce rilevata, registrazione in corso...")
                speech_started = True
                silent_chunks = 0
                frames.append(chunk)
            elif speech_started:
                frames.append(chunk)
                silent_chunks += 1
                seconds_of_silence = silent_chunks * chunk_duration
                print(f"🔇 Silenzio: {seconds_of_silence:.1f}s / {silence_duration}s", end="\r")
                if silent_chunks >= silence_chunks_needed:
                    break

    print("\n⏹ Registrazione terminata.")
    return np.concatenate(frames).flatten()


def transcribe() -> str:
    audio = listen()
    print("⏳ Trascrizione in corso...")
    result = whisper_model.transcribe(audio)
    print(f"🌍 Lingua rilevata: {result['language']}")
    text = result["text"].strip()
    print(f"📝 Hai detto: {text}")
    return text


if __name__ == "__main__":
    init_rag()

    with PostgresSaver.from_conn_string(get_db_address()) as memory:
        memory.setup()
        app = workflow.compile(checkpointer=memory)

        session_id = input("Inserire Id Sessione:\n")
        while True:
            raw = input("Inserire 1 per uscire, /ingest <file> per caricare documenti, /voice per parlare, oppure il tuo prompt:\n")

            if raw.strip() == "1":
                break

            if raw.strip().startswith("/ingest "):
                if len(raw.strip()) <= len("/ingest "):
                    print("[ERRORE] Comando /ingest richiede un percorso di file. Esempio: /ingest /path/to/file.pdf")
                else:
                    _handle_ingest(raw.strip()[len("/ingest "):])

            if raw.strip() == "/voice":
                prompt = transcribe()
            else:
                prompt = raw
            config = {"configurable": {"thread_id": session_id}}
            state = {
                "session_id": session_id,
                "original_text": prompt,
                "messages": [HumanMessage(content=prompt)],
            }
            output = app.invoke(state, config=config)
            response = output.get("response") or output["messages"][-1].content
            print(f"Risposta:\n{response}\n")
