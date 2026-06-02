import json
import re
import atexit
from langchain_core.tools import tool
from Shared.shared import engine, debug_print
from sqlalchemy import inspect, text
from Shared.shared import web_search
from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

# --- Sessione browser persistente ---

_playwright_instance = None
_browser: Optional[Browser] = None
_page: Optional[Page] = None

def _get_page() -> Page:
    global _playwright_instance, _browser, _page
    if _browser is None or not _browser.is_connected():
        debug_print("🌍 [BROWSER] Avvio nuova sessione browser...")
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=False)
        _page = _browser.new_context(storage_state=None).new_page()
    return _page

def _close_browser():
    global _playwright_instance, _browser, _page
    if _browser and _browser.is_connected():
        _browser.close()
    if _playwright_instance:
        _playwright_instance.stop()
    _browser = None
    _page = None
    _playwright_instance = None

atexit.register(_close_browser)

# --- Gestione popup / cookie banner ---

_COOKIE_SELECTORS = [
    # testo pulsante
    "button:has-text('Accetta')", "button:has-text('Accetta tutto')",
    "button:has-text('Accept')", "button:has-text('Accept all')",
    "button:has-text('Agree')", "button:has-text('I agree')",
    "button:has-text('OK')", "button:has-text('Got it')",
    "button:has-text('Consent')", "button:has-text('Allow all')",
    "button:has-text('Allow cookies')", "button:has-text('Continua')",
    # ID / class comuni
    "#accept-cookies", "#acceptCookies", "#cookie-accept",
    ".accept-cookies", ".cookie-accept", ".cookie-consent-accept",
    "[data-testid='cookie-accept']", "[aria-label='Accept cookies']",
    # OneTrust / Cookiebot / Quantcast
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    ".qc-cmp2-summary-buttons button:first-child",
]

def _dismiss_popups(page) -> None:
    for selector in _COOKIE_SELECTORS:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=500):
                btn.click(timeout=1000)
                debug_print(f"   [BROWSER] Cookie banner chiuso con selettore: {selector}")
                page.wait_for_timeout(800)
                return
        except Exception:
            continue

def _extract_text(page) -> str:
    for selector in ("main", "article", "#content", "#main-content", ".content", "body"):
        try:
            el = page.locator(selector).first
            if el.count() and el.is_visible(timeout=300):
                testo = el.inner_text(timeout=3000)
                if len(testo.strip()) > 200:
                    debug_print(f"   [BROWSER] Testo estratto da selettore: {selector}")
                    return testo
        except Exception:
            continue
    return page.inner_text("body")

# --- Protezioni browser ---

_ALLOWED_ACTIONS = {"fetch", "click", "fill", "get_inputs"}

_BLOCKED_URL_PATTERNS = [
    r"localhost", r"127\.0\.0\.1", r"192\.168\.", r"10\.\d+\.\d+\.\d+",
    r"169\.254\.", r"0\.0\.0\.0",
    r"/admin", r"/delete", r"/remove", r"/drop",
    r"payment", r"checkout", r"confirm-order",
]

def _is_url_safe(url: str) -> bool:
    url_lower = url.lower()
    return not any(re.search(p, url_lower) for p in _BLOCKED_URL_PATTERNS)

def _request_human_approval(action: str, url: str, selector: Optional[str], text_input: Optional[str]) -> bool:
    print(f"\n⚠️  [BROWSER] Azione interattiva richiesta dal modello:")
    print(f"   Azione   : {action.upper()}")
    print(f"   URL      : {url}")
    if selector:
        print(f"   Selettore: {selector}")
    if text_input:
        print(f"   Testo    : {text_input}")
    confirm = input("   Confermi l'esecuzione? (s/n): ").strip().lower()
    return confirm == "s"


def _make_tool_response(success: bool, receipt: str, summary: str, details: str) -> dict:
    return json.dumps({
        "success": success,
        "receipt": receipt,
        "summary": summary,
        "details": details
    }, ensure_ascii=False)

@tool
def elenco_tabelle_db() -> dict:
    """Usa questo strumento per ottenere l'elenco di tutte le tabelle presenti nel database.
    QUESTO STRUMENTO NON ACCETTA ALCUN PARAMETRO DI INPUT (lasciare gli argomenti vuoti {})."""
    debug_print("🔌 [TOOL PYTHON] Esecuzione di: elenco_tabelle_db()...")
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            receipt = "Nessuna tabella trovata nel database."
            return _make_tool_response(False, receipt, receipt, receipt)

        details = f"Tabelle trovate nel database: {tables}"
        summary = f"Sono presenti {len(tables)} tabelle: {tables}."
        receipt = "Elenco tabelle ottenuto correttamente."
        return _make_tool_response(True, receipt, summary, details)

    except Exception as e:
        error_message = f"Errore critico durante la lettura delle tabelle: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)

@tool
def Create_table(table_name: str, columns: List[Dict[str, Any]]) -> dict:
    """Usa questo strumento per creare una nuova tabella nel database.
    
    PARAMETRI:
    - table_name: Il nome della tabella (es. 'utenti').
    - columns: Una lista di dizionari Python contenente le colonne. 
      Ogni dizionario deve avere le chiavi chiare 'name' e 'type' (es. 'constraint' opzionale).
      Esempio: [{"name": "id", "type": "INTEGER", "constraint": "PRIMARY KEY"}]
    """
    debug_print(f"🔌 [TOOL PYTHON] Esecuzione di: Create_table(table_name: {table_name})...")
    
    try:
        if not isinstance(columns, list) or not columns:
            receipt = "Il parametro 'columns' deve essere una lista valida e non vuota di colonne."
            return _make_tool_response(False, receipt, receipt, receipt)

        clm = []
        for col in columns:
            if not isinstance(col, dict):
                receipt = "Ogni elemento di 'columns' deve essere un dizionario valido."
                return _make_tool_response(False, receipt, receipt, receipt)
            name = col.get("name")
            col_type = col.get("type")
            constraint = col.get("constraint", "")
            
            if not name or not col_type:
                receipt = "Ogni colonna deve contenere obbligatoriamente i campi 'name' e 'type'."
                return _make_tool_response(False, receipt, receipt, receipt)
                
            clm.append(f"{name} {col_type} {constraint}".strip())

        column_definitions = ", ".join(clm)

        with engine.connect() as connection:
            final_query = f"CREATE TABLE {table_name} ({column_definitions});"
            debug_print(f"   [LOG TOOL] Query SQL generata: {final_query}\n")

            with connection.begin():
                connection.execute(text(final_query))

        summary = f"Creata tabella '{table_name}' con colonne: {', '.join([col['name'] for col in columns])}."
        details = f"Eseguita query SQL: {final_query}"
        receipt = f"Tabella '{table_name}' creata con successo."
        return _make_tool_response(True, receipt, summary, details)
        
    except Exception as e:    
        error_message = f"Errore critico durante la creazione della tabella: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)

    
@tool
def Find_table_info(table_name: str) -> dict:
    """Usa questo strumento per ottenere la struttura dettagliata di una tabella specifica (colonne, tipi di dato e vincoli)."""
    debug_print(f"🔌 [TOOL PYTHON] Esecuzione di: Find_table_info(table_name: {table_name})...")

    try:
        with engine.connect() as connection: 
            inspector = inspect(connection)
            
            existing_tables = inspector.get_table_names()
            if table_name not in existing_tables:
                receipt = f"Tabella '{table_name}' non trovata nel database."
                details = f"Tabelle disponibili: {existing_tables}"
                return _make_tool_response(False, receipt, receipt, details)

            columns_info = inspector.get_columns(table_name)
            if not columns_info:
                receipt = f"Nessuna colonna trovata per la tabella '{table_name}'."
                return _make_tool_response(False, receipt, receipt, receipt)

            details = f"Informazioni sulla tabella '{table_name}':\n"
            for col in columns_info:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                details += f"- Campo: {col['name']}, Tipo: {col['type']}, Opzioni: {nullable}\n"
            summary = f"La tabella '{table_name}' contiene {len(columns_info)} colonne."
            receipt = f"Struttura della tabella '{table_name}' recuperata correttamente."
            return _make_tool_response(True, receipt, summary, details)
            
    except Exception as e:
        error_message = f"Errore critico durante la lettura della struttura della tabella: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)
    
@tool
def Cerca_su_Web(query: str) -> dict:
    """Usa questo strumento SOLO quando l'utente chiede informazioni in tempo reale, 
    notizie recenti, fatti di attualità o concetti generali non presenti nel database aziendale.
    Il parametro 'query' deve essere la stringa testuale esatta da inviare al motore di ricerca (es. 'Meteo Milano oggi')."""
    debug_print(f"🌐 [TOOL WEB] Ricerca online in corso per la query: '{query}'...")
    
    try:
        risultati = web_search.invoke({"query": query})
        
        if not risultati:
            receipt = "La ricerca online non ha prodotto nessun risultato utile per questa query."
            return _make_tool_response(False, receipt, receipt, receipt)
            
        details = f"Risultati estratti dal Web per '{query}':\n\n{risultati}"
        summary = f"Ricerca web completata con successo per '{query}'. Contenuto trovato disponibile nei dettagli."
        receipt = f"Ricerca web eseguita con successo per '{query}'."
        return _make_tool_response(True, receipt, summary, details)
        
    except Exception as e:
        error_message = f"Errore durante la ricerca web: {str(e)}"
        return _make_tool_response(False, error_message, error_message, error_message)


@tool
def Interagisci_con_Pagina_Web(
    url: str,
    action: str,
    selector: Optional[str] = None,
    text_input: Optional[str] = None
) -> dict:
    """Usa questo strumento per interagire con una pagina web tramite browser reale (Playwright).
    Supporta quattro azioni:
    - 'fetch': carica la pagina e restituisce il testo visibile. Non richiede selector né text_input.
    - 'get_inputs': restituisce la lista di tutti gli elementi interattivi visibili nella pagina corrente
      (input, textarea, select, button) con il loro CSS selector, tipo, placeholder e label.
      Usalo PRIMA di 'fill' o 'click' per scoprire il selector corretto da usare.
    - 'click': clicca su un elemento identificato da un CSS selector.
    - 'fill': compila un campo di input (selector) con il valore text_input e invia il form.

    PARAMETRI:
    - url: URL completo della pagina (es. 'https://www.example.com').
    - action: una tra 'fetch', 'get_inputs', 'click', 'fill'.
    - selector: (obbligatorio per click/fill) CSS selector dell'elemento target (es. 'button#submit', 'input[name=q]').
    - text_input: (obbligatorio per fill) testo da inserire nel campo.

    QUANDO USARE QUESTO TOOL vs Cerca_su_Web:
    - Usa Cerca_su_Web per ricerche generali su DuckDuckGo.
    - Usa questo tool quando hai già un URL specifico da visitare, quando la pagina richiede
      JavaScript per caricare i dati, o quando devi interagire con elementi della pagina."""

    debug_print(f"🌍 [TOOL BROWSER] Azione '{action}' su URL: {url}")

    # --- Protezione 1: azione nella whitelist ---
    if action not in _ALLOWED_ACTIONS:
        msg = f"[ERROR]: Azione '{action}' non consentita. Azioni valide: {sorted(_ALLOWED_ACTIONS)}."
        return _make_tool_response(False, msg, msg, msg)

    # --- Protezione 2: URL sicuro ---
    if not _is_url_safe(url):
        msg = f"[ERROR]: URL '{url}' bloccato dalle policy di sicurezza (indirizzo locale o percorso vietato)."
        return _make_tool_response(False, msg, msg, msg)

    # --- Protezione 3: parametri obbligatori per azioni interattive ---
    if action == "click" and not selector:
        msg = "[ERROR]: L'azione 'click' richiede il parametro 'selector'."
        return _make_tool_response(False, msg, msg, msg)
    if action == "fill" and (not selector or not text_input):
        msg = "[ERROR]: L'azione 'fill' richiede sia 'selector' che 'text_input'."
        return _make_tool_response(False, msg, msg, msg)

    # --- Protezione 4: human-in-the-loop per azioni non read-only ---
    if action in ("click", "fill"):
        approved = _request_human_approval(action, url, selector, text_input)
        if not approved:
            msg = "[ERROR]: Azione annullata dall'utente."
            debug_print(f"   [LOG BROWSER] {msg}")
            return _make_tool_response(False, msg, msg, msg)

    try:
        page = _get_page()
        current_url = page.url
        if not current_url.startswith(url) and not url.startswith(current_url.rstrip("/")):
            debug_print(f"   [LOG BROWSER] Navigazione verso '{url}' (URL corrente: '{current_url}')")
            try:
                page.goto(url, timeout=15000, wait_until="domcontentloaded")
            except PlaywrightTimeoutError:
                debug_print(f"   [LOG BROWSER] Timeout con 'domcontentloaded', riprovo con 'load'...")
                page.goto(url, timeout=20000, wait_until="load")
        else:
            debug_print(f"   [LOG BROWSER] Già su '{current_url}', salto la navigazione.")

        if action == "get_inputs":
            page.wait_for_timeout(800)
            _dismiss_popups(page)
            elementi = page.evaluate("""() => {
                const selectors = 'input, textarea, select, button[type="submit"], button[type="button"]';
                return Array.from(document.querySelectorAll(selectors))
                    .filter(el => {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.height > 0;
                    })
                    .map(el => {
                        const parts = [];
                        if (el.tagName) parts.push(el.tagName.toLowerCase());
                        if (el.id) parts.push('#' + el.id);
                        else if (el.name) parts.push('[name=' + el.name + ']');
                        else if (el.type && el.type !== 'text') parts.push('[type=' + el.type + ']');
                        return {
                            selector: parts.join(''),
                            type: el.type || el.tagName.toLowerCase(),
                            placeholder: el.placeholder || '',
                            label: el.getAttribute('aria-label') || el.getAttribute('title') || '',
                            value: el.value || ''
                        };
                    })
                    .slice(0, 30);
            }""")

            if not elementi:
                receipt = "Nessun elemento interattivo trovato nella pagina corrente."
                return _make_tool_response(False, receipt, receipt, receipt)

            righe = [f"- selector: `{e['selector']}` | type: {e['type']} | placeholder: \"{e['placeholder']}\" | label: \"{e['label']}\"" for e in elementi]
            details = "Elementi interattivi trovati nella pagina:\n" + "\n".join(righe)
            summary = f"Trovati {len(elementi)} elementi interattivi nella pagina."
            receipt = "Elementi interattivi recuperati con successo."
            return _make_tool_response(True, receipt, summary, details)

        if action == "fetch":
            page.wait_for_timeout(1500)
            _dismiss_popups(page)
            testo = _extract_text(page)
            testo = re.sub(r'\n{3,}', '\n\n', testo).strip()
            testo_troncato = testo[:4000] + ("..." if len(testo) > 4000 else "")

            summary = f"Pagina '{url}' caricata correttamente. Estratti {len(testo)} caratteri di testo."
            details = f"Contenuto della pagina '{url}':\n\n{testo_troncato}"
            receipt = f"Pagina '{url}' recuperata con successo."
            input("\n   [BROWSER] Premi Invio per continuare...")
            return _make_tool_response(True, receipt, summary, details)

        elif action == "click":
            debug_print(f"[LOG BROWSER] Clicco sul selettore '{selector}'...")
            page.locator(selector).first.click(timeout=8000)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PlaywrightTimeoutError:
                page.wait_for_timeout(2000)
            _dismiss_popups(page)
            testo = _extract_text(page)
            testo_troncato = testo[:4000] + ("..." if len(testo) > 4000 else "")

            summary = f"Click su '{selector}' eseguito. Pagina risultante estratta."
            details = f"Contenuto della pagina dopo il click:\n\n{testo_troncato}"
            receipt = f"Click su '{selector}' eseguito con successo."
            input("\n   [BROWSER] Premi Invio per continuare...")
            return _make_tool_response(True, receipt, summary, details)

        elif action == "fill":
            debug_print(f"[LOG BROWSER] Compilazione del campo '{selector}' con il testo '{text_input}'...")
            page.locator(selector).first.fill(text_input, timeout=8000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(2000)
            _dismiss_popups(page)
            testo = _extract_text(page)
            testo_troncato = testo[:4000] + ("..." if len(testo) > 4000 else "")

            summary = f"Campo '{selector}' compilato con '{text_input}' e form inviato."
            details = f"Contenuto della pagina dopo l'invio del form:\n\n{testo_troncato}"
            receipt = f"Form compilato e inviato con successo su '{url}'."
            input("\n   [BROWSER] Premi Invio per continuare...")
            return _make_tool_response(True, receipt, summary, details)

    except PlaywrightTimeoutError as e:
        msg = f"[ERROR]: Timeout durante l'interazione con '{url}': {str(e)}"
        return _make_tool_response(False, msg, msg, msg)
    except Exception as e:
        msg = f"[ERROR]: Errore durante l'interazione con il browser: {str(e)}"
        return _make_tool_response(False, msg, msg, msg)