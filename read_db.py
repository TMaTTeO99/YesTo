import json
from Shared.shared import getDBAddress
from langgraph.checkpoint.postgres import PostgresSaver

def inspect_thread(thread_id: str):
    # Ci colleghiamo al database PostgresSaver
    with PostgresSaver.from_conn_string(getDBAddress()) as memory:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Recuperiamo l'ultimo checkpoint salvato per questo thread
        checkpoint_tuple = memory.get_tuple(config)
        
        if not checkpoint_tuple:
            print(f"❌ Nessun dato trovato nel DB per il Thread ID: '{thread_id}'")
            return
            
        print(f"\n===== 🗄️ ISPEZIONE DATABASE POSTGRES (Thread: {thread_id}) =====")
        
        # Lo stato attuale delle variabili personalizzate (threshold, critiche, ecc.)
        state_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        
        print("\n📊 VARIABILI DI STATO ATTUALI:")
        print(f"  • Threshold (Tentativi): {state_values.get('threshold', 0)}")
        print(f"  • Critique Approvata: {state_values.get('critique_approvata', False)}")
        print(f"  • Critique Punti (Note del supervisore): {state_values.get('critique_punti', [])}")
        print(f"  • Original Text (Ultimo input): '{state_values.get('original_text', '')}'")
        
        print("\n💬 CRONOLOGIA MESSAGGI (chat_history):")
        # I messaggi sono serializzati dentro lo stato, li estraiamo ordinati
        messages = state_values.get("messages", [])
        if not messages:
            print("  [Nessun messaggio presente]")
        else:
            for idx, msg in enumerate(messages, 1):
                # Capiamo il tipo di messaggio (Human o AI)
                msg_type = msg.__class__.__name__
                print(f"  [{idx}] {msg_type}: {msg.content}")
                
        print("\n======================================================\n")

if __name__ == "__main__":
    target_id = input("Inserire l'ID Sessione (thread_id) da ispezionare:\n")
    inspect_thread(thread_id=target_id)