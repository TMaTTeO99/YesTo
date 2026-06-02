import json
import re
import atexit
from typing import Optional
from langchain_core.tools import tool
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from config import debug_print


# ---------------------------------------------------------------------------
# Persistent browser session — one Chromium instance for the entire process.
# ---------------------------------------------------------------------------

_playwright_instance = None
_browser: Optional[Browser] = None
_page: Optional[Page] = None


def _get_page() -> Page:
    global _playwright_instance, _browser, _page
    if _browser is None or not _browser.is_connected():
        debug_print("🌍 [BROWSER] Avvio nuova sessione browser...")
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=False)
        _page = _browser.new_context().new_page()
    return _page


def _close_browser():
    global _playwright_instance, _browser, _page
    if _browser and _browser.is_connected():
        _browser.close()
    if _playwright_instance:
        _playwright_instance.stop()
    _browser = _page = _playwright_instance = None


atexit.register(_close_browser)


# ---------------------------------------------------------------------------
# Cookie / popup dismissal
# ---------------------------------------------------------------------------

_COOKIE_SELECTORS = [
    "button:has-text('Accetta')", "button:has-text('Accetta tutto')",
    "button:has-text('Accept')", "button:has-text('Accept all')",
    "button:has-text('Agree')", "button:has-text('I agree')",
    "button:has-text('OK')", "button:has-text('Got it')",
    "button:has-text('Consent')", "button:has-text('Allow all')",
    "button:has-text('Allow cookies')", "button:has-text('Continua')",
    "#accept-cookies", "#acceptCookies", "#cookie-accept",
    ".accept-cookies", ".cookie-accept", ".cookie-consent-accept",
    "[data-testid='cookie-accept']", "[aria-label='Accept cookies']",
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    ".qc-cmp2-summary-buttons button:first-child",
]


def _dismiss_popups(page: Page) -> None:
    for selector in _COOKIE_SELECTORS:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=500):
                btn.click(timeout=1000)
                debug_print(f"   [BROWSER] Cookie banner chiuso: {selector}")
                page.wait_for_timeout(800)
                return
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Smart text extraction — prefers semantic containers over raw <body>
# ---------------------------------------------------------------------------

def _extract_text(page: Page) -> str:
    for selector in ("main", "article", "#content", "#main-content", ".content", "body"):
        try:
            el = page.locator(selector).first
            if el.count() and el.is_visible(timeout=300):
                text = el.inner_text(timeout=3000)
                if len(text.strip()) > 200:
                    debug_print(f"   [BROWSER] Testo estratto da: {selector}")
                    return text
        except Exception:
            continue
    return page.inner_text("body")


# ---------------------------------------------------------------------------
# Security guards
# ---------------------------------------------------------------------------

_ALLOWED_ACTIONS = {"fetch", "get_inputs", "click", "fill"}

_BLOCKED_URL_PATTERNS = [
    r"localhost", r"127\.0\.0\.1", r"192\.168\.", r"10\.\d+\.\d+\.\d+",
    r"169\.254\.", r"0\.0\.0\.0",
    r"/admin", r"/delete", r"/remove", r"/drop",
    r"payment", r"checkout", r"confirm-order",
]


def _is_url_safe(url: str) -> bool:
    return not any(re.search(p, url.lower()) for p in _BLOCKED_URL_PATTERNS)


def _request_human_approval(action: str, url: str, selector: Optional[str], text_input: Optional[str]) -> bool:
    print(f"\n⚠️  [BROWSER] Azione interattiva richiesta dal modello:")
    print(f"   Azione   : {action.upper()}")
    print(f"   URL      : {url}")
    if selector:
        print(f"   Selettore: {selector}")
    if text_input:
        print(f"   Testo    : {text_input}")
    return input("   Confermi l'esecuzione? (s/n): ").strip().lower() == "s"


# ---------------------------------------------------------------------------
# Shared response builder
# ---------------------------------------------------------------------------

def _make_tool_response(success: bool, receipt: str, summary: str, details: str) -> str:
    return json.dumps(
        {"success": success, "receipt": receipt, "summary": summary, "details": details},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def browse_page(
    url: str,
    action: str,
    selector: Optional[str] = None,
    text_input: Optional[str] = None,
) -> str:
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
    - selector: (obbligatorio per click/fill) CSS selector dell'elemento target.
    - text_input: (obbligatorio per fill) testo da inserire nel campo.

    QUANDO USARE QUESTO TOOL vs web_search_tool:
    - Usa web_search_tool per ricerche generali su DuckDuckGo.
    - Usa questo tool quando hai già un URL specifico, la pagina richiede JavaScript,
      o devi interagire con elementi della pagina.
    - Per digitare testo in un campo: prima usa 'get_inputs' per trovare il selector,
      poi usa 'fill' con il selector esatto trovato. NON indovinare il selector."""

    debug_print(f"🌍 [TOOL BROWSER] browse_page: action='{action}' url='{url}'")

    if action not in _ALLOWED_ACTIONS:
        msg = f"[ERROR]: Azione '{action}' non consentita. Valide: {sorted(_ALLOWED_ACTIONS)}."
        return _make_tool_response(False, msg, msg, msg)

    if not _is_url_safe(url):
        msg = f"[ERROR]: URL '{url}' bloccato dalle policy di sicurezza."
        return _make_tool_response(False, msg, msg, msg)

    if action == "click" and not selector:
        msg = "[ERROR]: L'azione 'click' richiede il parametro 'selector'."
        return _make_tool_response(False, msg, msg, msg)

    if action == "fill" and (not selector or not text_input):
        msg = "[ERROR]: L'azione 'fill' richiede sia 'selector' che 'text_input'."
        return _make_tool_response(False, msg, msg, msg)

    if action in ("click", "fill"):
        if not _request_human_approval(action, url, selector, text_input):
            msg = "[ERROR]: Azione annullata dall'utente."
            return _make_tool_response(False, msg, msg, msg)

    try:
        page = _get_page()
        current_url = page.url

        if not current_url.startswith(url) and not url.startswith(current_url.rstrip("/")):
            debug_print(f"   [BROWSER] Navigazione verso '{url}'")
            try:
                page.goto(url, timeout=15000, wait_until="domcontentloaded")
            except PlaywrightTimeoutError:
                debug_print("   [BROWSER] Timeout con 'domcontentloaded', riprovo con 'load'...")
                page.goto(url, timeout=20000, wait_until="load")
        else:
            debug_print(f"   [BROWSER] Già su '{current_url}', salto la navigazione.")

        if action == "get_inputs":
            page.wait_for_timeout(800)
            _dismiss_popups(page)
            elements = page.evaluate("""() => {
                const sel = 'input, textarea, select, button[type="submit"], button[type="button"]';
                return Array.from(document.querySelectorAll(sel))
                    .filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
                    .map(el => {
                        const parts = [el.tagName.toLowerCase()];
                        if (el.id) parts.push('#' + el.id);
                        else if (el.name) parts.push('[name=' + el.name + ']');
                        else if (el.type && el.type !== 'text') parts.push('[type=' + el.type + ']');
                        return {
                            selector: parts.join(''),
                            type: el.type || el.tagName.toLowerCase(),
                            placeholder: el.placeholder || '',
                            label: el.getAttribute('aria-label') || el.getAttribute('title') || '',
                        };
                    }).slice(0, 30);
            }""")

            if not elements:
                msg = "Nessun elemento interattivo trovato nella pagina corrente."
                return _make_tool_response(False, msg, msg, msg)

            rows = [
                f"- selector: `{e['selector']}` | type: {e['type']} | placeholder: \"{e['placeholder']}\" | label: \"{e['label']}\""
                for e in elements
            ]
            details = "Elementi interattivi trovati nella pagina:\n" + "\n".join(rows)
            summary = f"Trovati {len(elements)} elementi interattivi nella pagina."
            return _make_tool_response(True, "Elementi interattivi recuperati con successo.", summary, details)

        if action == "fetch":
            page.wait_for_timeout(1500)
            _dismiss_popups(page)
            text = _extract_text(page)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            truncated = text[:4000] + ("..." if len(text) > 4000 else "")
            summary = f"Pagina '{url}' caricata. Estratti {len(text)} caratteri."
            details = f"Contenuto della pagina '{url}':\n\n{truncated}"
            input("\n   [BROWSER] Premi Invio per continuare...")
            return _make_tool_response(True, f"Pagina '{url}' recuperata con successo.", summary, details)

        if action == "click":
            debug_print(f"   [BROWSER] Click su '{selector}'")
            page.locator(selector).first.click(timeout=8000)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PlaywrightTimeoutError:
                page.wait_for_timeout(2000)
            _dismiss_popups(page)
            text = _extract_text(page)
            truncated = text[:4000] + ("..." if len(text) > 4000 else "")
            input("\n   [BROWSER] Premi Invio per continuare...")
            return _make_tool_response(
                True, f"Click su '{selector}' eseguito con successo.",
                f"Click su '{selector}' eseguito. Pagina risultante estratta.",
                f"Contenuto della pagina dopo il click:\n\n{truncated}",
            )

        if action == "fill":
            debug_print(f"   [BROWSER] Fill '{selector}' con '{text_input}'")
            page.locator(selector).first.fill(text_input, timeout=8000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(2000)
            _dismiss_popups(page)
            text = _extract_text(page)
            truncated = text[:4000] + ("..." if len(text) > 4000 else "")
            input("\n   [BROWSER] Premi Invio per continuare...")
            return _make_tool_response(
                True, f"Form compilato e inviato su '{url}'.",
                f"Campo '{selector}' compilato con '{text_input}' e form inviato.",
                f"Contenuto della pagina dopo l'invio:\n\n{truncated}",
            )

    except PlaywrightTimeoutError as e:
        msg = f"[ERROR]: Timeout durante l'interazione con '{url}': {e}"
        return _make_tool_response(False, msg, msg, msg)
    except Exception as e:
        msg = f"[ERROR]: Errore durante l'interazione con il browser: {e}"
        return _make_tool_response(False, msg, msg, msg)
