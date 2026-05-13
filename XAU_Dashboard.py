import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import dash_bootstrap_components as dbc
from datetime import datetime
import calendar
import logging
from functools import lru_cache

# ==========================================
# 0. KONFIGURACJA LOGÓW I BIBLIOTEK
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XAU_QUANT")

try:
    import cot_reports as cot
except ImportError:
    cot = None
    logger.warning("Brak biblioteki 'cot_reports'. Dane COT nie będą pobierane (użyj: pip install cot_reports).")

# ==========================================
# 1. DESIGN SYSTEM (INSTITUTIONAL QUANT)
# ==========================================
COLORS = {
    'bg': '#000000',          
    'panel': '#0a0b10',       
    'border': '#1a1d24',      
    'gold': '#d4af37',        
    'blue': '#1e90ff',        
    'text_main': '#ffffff',   
    'text_sub': '#8b9bb4',    
    'up': '#10b981',          
    'down': '#ef4444',
    'neutral': '#64748b'
}

CARD_STYLE = {
    'backgroundColor': COLORS['panel'],
    'border': f"1px solid {COLORS['border']}",
    'borderRadius': '8px',
    'boxShadow': '0 6px 16px rgba(0, 0, 0, 0.5)'
}

MONO = {'fontFamily': "'Consolas', 'Courier New', monospace", 'letterSpacing': '0.5px'}
LABEL_STYLE = {'color': COLORS['text_sub'], 'fontSize': '0.9rem', 'fontWeight': '600', 'letterSpacing': '1px', 'textTransform': 'uppercase'}

BASE_CHART_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=15, r=15, t=35, b=15),
    hovermode="x unified",
    font=dict(color=COLORS['text_sub'], family="'Consolas', monospace", size=12),
    hoverlabel=dict(font_size=14, font_family="'Consolas', monospace")
)

# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================
def format_num(val):
    if pd.isna(val): return "0"
    return f"{val:,.0f}".replace(",", " ")

def get_color(val):
    if pd.isna(val) or val == 0: return COLORS['text_main']
    return COLORS['up'] if val > 0 else COLORS['down']

def render_cot_cell(val, chg, is_net=False):
    val = 0 if pd.isna(val) else val
    chg = 0 if pd.isna(chg) else chg
    val_color = get_color(val) if is_net else COLORS['text_main']
    chg_color = COLORS['up'] if chg > 0 else (COLORS['down'] if chg < 0 else COLORS['text_sub'])
    sign = "+" if chg > 0 else ""
    return html.Td([
        html.Div(format_num(val), style={'color': val_color, 'fontWeight': 'bold' if is_net else 'normal', 'fontSize': '1.1rem'}),
        html.Div(f"{sign}{format_num(chg)}", style={'fontSize': '0.85rem', 'color': chg_color, 'marginTop': '2px'})
    ], style=dict(MONO, textAlign='right', border='none', verticalAlign='middle', padding='0.75rem'))

# ==========================================
# 3. WARSTWA DANYCH (DATA LAYER)
# ==========================================
@lru_cache(maxsize=1)
def fetch_live_gold_data(ticker='GC=F', lookback_days=5):
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return float(df['Open'].dropna().iloc[-1])
    except Exception as e:
        logger.error(f"Błąd live gold: {e}")
        return None

@lru_cache(maxsize=1)
def fetch_intermarket_history(lookback='1y'):
    try:
        df_gold = yf.download('GC=F', period=lookback, interval="1d", progress=False)
        if isinstance(df_gold.columns, pd.MultiIndex): df_gold.columns = df_gold.columns.droplevel(1)
        df_bonds = yf.download('ZN=F', period=lookback, interval="1d", progress=False)
        if isinstance(df_bonds.columns, pd.MultiIndex): df_bonds.columns = df_bonds.columns.droplevel(1)
        return pd.DataFrame({'Gold': df_gold['Close'], 'Bonds': df_bonds['Close']}).dropna()
    except Exception as e:
        logger.error(f"Błąd intermarket data: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=2)
def fetch_cot_data(commodity_label="GOLD"):
    if cot is None: return None, None
    try:
        current_year = datetime.now().year
        df_cot = cot.cot_year(year=current_year, cot_report_type='legacy_fut')
        if df_cot is None or df_cot.empty: return None, None

        df_cot.columns = df_cot.columns.str.strip().str.replace(' ', '_')
        market_col = next((c for c in df_cot.columns if 'Market' in c and 'Exchange' in c), None) or [c for c in df_cot.columns if 'Name' in c][0]
        df_asset = df_cot[df_cot[market_col].astype(str).str.contains(commodity_label, case=False, na=False)].copy()
        if df_asset.empty: return None, None

        date_col = next((c for c in df_asset.columns if 'Date' in c or 'date' in c), None)
        df_asset['Parsed_Date'] = pd.to_datetime(df_asset[date_col].astype(str).str.zfill(6), format='%y%m%d', errors='coerce') if 'YYMMDD' in date_col else pd.to_datetime(df_asset[date_col], errors='coerce')
        df_asset = df_asset.dropna(subset=['Parsed_Date']).sort_values('Parsed_Date')

        def get_col(keywords, exclude=[]):
            return next((col for col in df_asset.columns if all(k in col.lower().replace('_', '').replace('-', '') for k in keywords) and not any(e in col.lower().replace('_', '').replace('-', '') for e in exclude)), None)

        cols = {
            'c_l': get_col(['comm', 'long'], ['non']), 'c_s': get_col(['comm', 'short'], ['non']),
            'nc_l': get_col(['noncomm', 'long']), 'nc_s': get_col(['noncomm', 'short']),
            's_l': get_col(['non', 'rep', 'long'], ['comm']) or get_col(['small', 'long']), 's_s': get_col(['non', 'rep', 'short'], ['comm']) or get_col(['small', 'short']),
            'oi': get_col(['open', 'interest']) or get_col(['oi'])
        }

        def calc_net(l, s): return pd.to_numeric(df_asset[l].fillna(0)) - pd.to_numeric(df_asset[s].fillna(0)) if l and s else pd.Series([0]*len(df_asset), index=df_asset.index)
        df_asset['Comm_Net'] = calc_net(cols['c_l'], cols['c_s'])
        df_asset['NonComm_Net'] = calc_net(cols['nc_l'], cols['nc_s'])
        df_asset['Small_Net'] = calc_net(cols['s_l'], cols['s_s'])

        def calc_index(series, w=26):
            diff = series.rolling(w, min_periods=1).max() - series.rolling(w, min_periods=1).min()
            return np.where(diff == 0, 50, (series - series.rolling(w, min_periods=1).min()) / diff * 100)

        for p in ['Comm', 'NonComm', 'Small']: df_asset[f'{p}_Index'] = calc_index(df_asset[f'{p}_Net'])

        valid_rows = df_asset.dropna(subset=['Comm_Index'])
        if valid_rows.empty: return None, None
        latest = valid_rows.iloc[-1]
        prev = valid_rows.iloc[-2] if len(valid_rows) >= 2 else latest

        def s_get(row, c): return row[c] if c and c in row else 0

        return {
            'Date': latest['Parsed_Date'].strftime('%Y-%m-%d') if pd.notna(latest['Parsed_Date']) else '---',
            'OI': s_get(latest, cols['oi']), 'OI_Chg': s_get(latest, cols['oi']) - s_get(prev, cols['oi']),
            'Comm_Long': s_get(latest, cols['c_l']), 'Comm_Long_Chg': s_get(latest, cols['c_l']) - s_get(prev, cols['c_l']),
            'Comm_Short': s_get(latest, cols['c_s']), 'Comm_Short_Chg': s_get(latest, cols['c_s']) - s_get(prev, cols['c_s']),
            'Comm_Net': latest['Comm_Net'], 'Comm_Net_Chg': latest['Comm_Net'] - prev['Comm_Net'], 'Comm_Index': latest['Comm_Index'],
            'NonComm_Long': s_get(latest, cols['nc_l']), 'NonComm_Long_Chg': s_get(latest, cols['nc_l']) - s_get(prev, cols['nc_l']),
            'NonComm_Short': s_get(latest, cols['nc_s']), 'NonComm_Short_Chg': s_get(latest, cols['nc_s']) - s_get(prev, cols['nc_s']),
            'NonComm_Net': latest['NonComm_Net'], 'NonComm_Net_Chg': latest['NonComm_Net'] - prev['NonComm_Net'], 'NonComm_Index': latest['NonComm_Index'],
            'Small_Long': s_get(latest, cols['s_l']), 'Small_Long_Chg': s_get(latest, cols['s_l']) - s_get(prev, cols['s_l']),
            'Small_Short': s_get(latest, cols['s_s']), 'Small_Short_Chg': s_get(latest, cols['s_s']) - s_get(prev, cols['s_s']),
            'Small_Net': latest['Small_Net'], 'Small_Net_Chg': latest['Small_Net'] - prev['Small_Net'], 'Small_Index': latest['Small_Index']
        }, df_asset
    except Exception as e:
        logger.error(f"Błąd przetwarzania COT: {e}")
        return None, None

@lru_cache(maxsize=1)
def load_daily_data(filepath='XAU_1d_data.csv'):
    try:
        df = pd.read_csv(filepath, sep=';', index_col=0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df['Return_Points'] = df['Close'] - df['Open']
        df['MFE_Pct'] = ((df['High'] - df['Open']) / df['Open']) * 100
        df['MAE_Pct'] = ((df['Open'] - df['Low']) / df['Open']) * 100 
        df['Month'], df['Year'] = df.index.month, df.index.year
        df['YearMonth'] = df.index.to_period('M')
        df['TDOM'] = df.groupby('YearMonth').cumcount() + 1
        df['Weekday'], df['DayOfYear'], df['DateOnly'] = df.index.dayofweek, df.index.dayofyear, df.index.date
        df['HH_14'], df['LL_14'] = df['High'].rolling(window=14).max(), df['Low'].rolling(window=14).min()
        df['Williams_%R'] = ((df['HH_14'] - df['Close']) / (df['HH_14'] - df['LL_14'])) * -100
        return df
    except Exception as e:
        logger.error(f"Błąd {filepath}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=1)
def load_hourly_data(filepath='XAU_1h_data.csv'):
    try:
        df_h = pd.read_csv(filepath, sep=';')
        df_h['Date'] = pd.to_datetime(df_h['Date'], format='%Y.%m.%d %H:%M')
        df_h.set_index('Date', inplace=True)
        df_h['Hour'], df_h['DateOnly'] = df_h.index.hour, df_h.index.date
        df_h['Return_Points'] = df_h['Close'] - df_h['Open'] 
        return df_h
    except Exception as e:
        logger.error(f"Błąd {filepath}: {e}")
        return pd.DataFrame()

# ==========================================
# 4. INICJALIZACJA DANYCH
# ==========================================
logger.info("Ładowanie danych...")
df = load_daily_data()
df_hourly = load_hourly_data()
df_intermarket = fetch_intermarket_history()
live_open = fetch_live_gold_data()

cot_gold, df_asset_gold = fetch_cot_data("GOLD")
cot_usd, _ = fetch_cot_data("DOLLAR INDEX")

def safe_cot():
    return {k: (50 if 'Index' in k else 0) for k in ['OI', 'OI_Chg', 'Comm_Long', 'Comm_Long_Chg', 'Comm_Short', 'Comm_Short_Chg', 'Comm_Net', 'Comm_Net_Chg', 'Comm_Index', 'NonComm_Long', 'NonComm_Long_Chg', 'NonComm_Short', 'NonComm_Short_Chg', 'NonComm_Net', 'NonComm_Net_Chg', 'NonComm_Index', 'Small_Long', 'Small_Long_Chg', 'Small_Short', 'Small_Short_Chg', 'Small_Net', 'Small_Net_Chg', 'Small_Index']} | {'Date': '---'}

cot_gold = cot_gold or safe_cot()
cot_usd = cot_usd or safe_cot()

oi_change, price_change = 0, 0
if df_asset_gold is not None and not df_asset_gold.empty and len(df_asset_gold) >= 5:
    oi_col = next((c for c in df_asset_gold.columns if 'open' in c.lower() and 'interest' in c.lower()), None) or next((c for c in df_asset_gold.columns if c.lower() == 'oi'), None)
    if oi_col:
        try:
            oi_change = pd.to_numeric(df_asset_gold.iloc[-1][oi_col], errors='coerce') - pd.to_numeric(df_asset_gold.iloc[-5][oi_col], errors='coerce')
            past_idx = df.index.get_indexer([df_asset_gold.iloc[-5]['Parsed_Date'].replace(tzinfo=None)], method='nearest')[0]
            price_change = df.iloc[-1]['Close'] - df.iloc[past_idx]['Close']
        except Exception: pass

# --- AI QUANT NARRACJA ---
c_idx, c_net_chg, s_idx, oi_chg = cot_gold.get('Comm_Index', 50), cot_gold.get('Comm_Net_Chg', 0), cot_gold.get('Small_Index', 50), cot_gold.get('OI_Chg', 0)

# 1. Smart Money & Delta
if c_idx >= 75:
    comm_desc = html.Span([html.B(f"Smart Money silnie kupują (Index: {c_idx:.1f}%). ", style={'color': COLORS['up'], 'fontSize': '1.05rem'}), "Jesteśmy w strefie ekstremalnego niedowartościowania. To potężny, historyczny sygnał na zbliżające się WZROSTY."])
elif c_idx <= 25:
    comm_desc = html.Span([html.B(f"Smart Money silnie sprzedają (Index: {c_idx:.1f}%). ", style={'color': COLORS['down'], 'fontSize': '1.05rem'}), "Jesteśmy w strefie ekstremalnego przewartościowania. Kopalnie szykują się na SPADKI ceny złota."])
else:
    comm_desc = html.Span([html.B(f"Smart Money - Sentyment neutralny (Index: {c_idx:.1f}%). ", style={'color': COLORS['text_main'], 'fontSize': '1.05rem'}), "Brak skrajnego ułożenia kapitału. Oczekujemy na wyraźniejszy sygnał kierunkowy."])

if c_net_chg > 0:
    delta_desc = html.Span([html.B("Budowanie Byczej Presji: ", style={'color': COLORS['up']}), f"W ubiegłym tygodniu dołożyli do rynku netto {format_num(c_net_chg)} pozycji długich."])
elif c_net_chg < 0:
    delta_desc = html.Span([html.B("Budowanie Niedźwiedziej Presji: ", style={'color': COLORS['down']}), f"W ubiegłym tygodniu zredukowali rynek netto o {format_num(abs(c_net_chg))} pozycji (agresywne shortowanie)."])
else:
    delta_desc = html.Span("Brak znaczących zmian w pozycjonowaniu w ostatnim tygodniu.", style={'color': COLORS['text_sub']})

# 2. Ulica (Dumb Money)
if s_idx >= 75:
    small_desc = html.Span([html.B(f"Ulica w euforii (Index: {s_idx:.1f}%). ", style={'color': COLORS['down'], 'fontSize': '1.05rem'}), "Tłum agresywnie kupuje na górce. Złota zasada kontrariańska mówi: jeśli 'ulica' jest pewna wzrostów, szykuj się na brutalną korektę w dół."])
elif s_idx <= 25:
    small_desc = html.Span([html.B(f"Ulica w panice (Index: {s_idx:.1f}%). ", style={'color': COLORS['up'], 'fontSize': '1.05rem'}), "Drobni inwestorzy uciekają z rynku. Kontrariańsko, to często najlepszy moment, by dołączyć do 'grubych ryb' i szukać pozycji LONG."])
else:
    small_desc = html.Span([html.B(f"Ulica bez kierunku (Index: {s_idx:.1f}%). ", style={'color': COLORS['text_main'], 'fontSize': '1.05rem'}), "Drobni spekulanci nie przejawiają skrajnych emocji, wskaźnik znajduje się pośrodku."])

# 3. Open Interest i Momentum
if price_change > 0 and oi_chg > 0:
    oi_desc = html.Span([html.B("ZDROWA HOSSA (Silnik pełen paliwa). ", style={'color': COLORS['up'], 'fontSize': '1.05rem'}), "Cena rośnie przy jednoczesnym wzroście Open Interest. Na rynek wchodzi ogromny, świeży kapitał wspierający długoterminowe wzrosty."])
elif price_change > 0 and oi_chg < 0:
    oi_desc = html.Span([html.B("SHORT SQUEEZE (Paliwo się kończy). ", style={'color': COLORS['down'], 'fontSize': '1.05rem'}), "Cena rośnie, ale OI spada. Wzrosty wynikają głównie z panicznego uciekania sprzedających (zamykanie szortów). Słaby fundament do dalszych wzrostów."])
elif price_change < 0 and oi_chg > 0:
    oi_desc = html.Span([html.B("ZDROWA BESSA (Agresywna dystrybucja). ", style={'color': COLORS['down'], 'fontSize': '1.05rem'}), "Cena spada, a OI drastycznie rośnie. Grubi gracze wlewają na rynek nowy kapitał do grania na spadki."])
elif price_change < 0 and oi_chg < 0:
    oi_desc = html.Span([html.B("KAPITULACJA (Spadki tracą impet). ", style={'color': COLORS['up'], 'fontSize': '1.05rem'}), "Cena i OI spadają. Wyprzedaż ustaje, kapitał powoli ewakuuje się ze zyskownych szortów. Rynek szykuje się do uklepania mocnego dołka."])
else:
    oi_desc = html.Span([html.B("NEUTRALNIE. ", style={'color': COLORS['text_main'], 'fontSize': '1.05rem'}), "Cena i OI nie ukształtowały w ostatnim tygodniu wyraźnej, skorelowanej dynamiki."])

# ==========================================
# 5. KOMPONENTY UI (MODULARNE)
# ==========================================
def build_header():
    return dbc.Row([
        dbc.Col([
            html.H2([html.I(className="bi bi-hexagon-fill me-3", style={'color': COLORS['gold']}), "XAU QUANT PRO TERMINAL"], className="fw-bold m-0", style={'letterSpacing': '2px', 'color': COLORS['text_main']}),
            html.Div(f"System Status: ONLINE | Last Sync: {datetime.now().strftime('%H:%M:%S UTC')}", style={'color': COLORS['up'], 'fontSize': '1rem', 'marginTop': '8px', 'fontFamily': MONO['fontFamily']})
        ], width=12)
    ], className="border-bottom pb-4 mb-4 mt-3", style={'borderColor': COLORS['border']})

def build_cot_panel():
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([html.Span("RAPORT CFTC (COT) - ZŁOTO", style=LABEL_STYLE), html.Span(f"DATA: {cot_gold['Date']}", className="float-end", style=LABEL_STYLE)], className="mb-4 border-bottom pb-3", style={'borderColor': COLORS['border']}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("OPEN INTEREST: ", style={'fontSize': '1.1rem'}), 
                        html.Span(format_num(cot_gold['OI']), style={'fontSize': '1.2rem', 'fontWeight': 'bold'}), 
                        html.Span(f" ({'+' if cot_gold.get('OI_Chg', 0) > 0 else ''}{format_num(cot_gold.get('OI_Chg', 0))})", style={'color': COLORS['up'] if cot_gold.get('OI_Chg', 0) > 0 else (COLORS['down'] if cot_gold.get('OI_Chg', 0) < 0 else COLORS['text_sub']), 'fontSize': '1rem', 'marginLeft': '8px', 'fontWeight': 'bold'})
                    ], style=dict(LABEL_STYLE, color=COLORS['gold'], marginBottom='15px')),
                    dbc.Table([
                        html.Thead(html.Tr([html.Th(k, style=dict(LABEL_STYLE, borderBottom=f"2px solid {COLORS['border']}", textAlign='right' if i>0 else 'left')) for i, k in enumerate(["GRUPA", "LONG", "SHORT", "NETTO", "INDEX"])])),
                        html.Tbody([
                            html.Tr([html.Td("COMMERCIALS", style=dict(LABEL_STYLE, color=COLORS['text_main'], border='none', fontSize='1rem')), render_cot_cell(cot_gold['Comm_Long'], cot_gold['Comm_Long_Chg']), render_cot_cell(cot_gold['Comm_Short'], cot_gold['Comm_Short_Chg']), render_cot_cell(cot_gold['Comm_Net'], cot_gold['Comm_Net_Chg'], True), html.Td(f"{cot_gold['Comm_Index']:.1f}%", style=dict(MONO, color=COLORS['gold'], textAlign='right', border='none', fontSize='1.2rem', fontWeight='bold'))]),
                            html.Tr([html.Td("LARGE SPECS", style=dict(LABEL_STYLE, color=COLORS['text_main'], border='none', fontSize='1rem')), render_cot_cell(cot_gold['NonComm_Long'], cot_gold['NonComm_Long_Chg']), render_cot_cell(cot_gold['NonComm_Short'], cot_gold['NonComm_Short_Chg']), render_cot_cell(cot_gold['NonComm_Net'], cot_gold['NonComm_Net_Chg'], True), html.Td(f"{cot_gold['NonComm_Index']:.1f}%", style=dict(MONO, color=COLORS['gold'], textAlign='right', border='none', fontSize='1.2rem', fontWeight='bold'))]),
                            html.Tr([html.Td("SMALL TRADERS", style=dict(LABEL_STYLE, color=COLORS['text_main'], border='none', fontSize='1rem')), render_cot_cell(cot_gold['Small_Long'], cot_gold['Small_Long_Chg']), render_cot_cell(cot_gold['Small_Short'], cot_gold['Small_Short_Chg']), render_cot_cell(cot_gold['Small_Net'], cot_gold['Small_Net_Chg'], True), html.Td(f"{cot_gold['Small_Index']:.1f}%", style=dict(MONO, color=COLORS['gold'], textAlign='right', border='none', fontSize='1.2rem', fontWeight='bold'))])
                        ])
                    ], size="md", className="mb-0 table-borderless"),
                ], md=7, className="pe-4 border-end", style={'borderColor': COLORS['border']}),
                dbc.Col([html.Div("POZYCJE NETTO", style=dict(LABEL_STYLE, textAlign='center', marginBottom='10px')), dcc.Graph(id='cot-bar-chart', config={'displayModeBar': False}, style={'height': '180px'})], md=5, className="ps-4")
            ])
        ], className="p-4"), style=CARD_STYLE, className="mb-4"), width=12)
    ])

def build_synthesis_panel():
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(html.Span("SYNTEZA DANYCH (AI NARRATIVE)", style=LABEL_STYLE), className="mb-3 border-bottom pb-3", style={'borderColor': COLORS['border']}),
            dbc.Row([
                dbc.Col([html.H5("SMART MONEY", className="fw-bold mb-3", style={'color': COLORS['gold']}), html.P(comm_desc, style={'fontSize': '1rem', 'marginBottom': '8px', 'color': COLORS['text_sub'], 'lineHeight': '1.5'}), html.P(delta_desc, style={'fontSize': '1rem', 'marginBottom': '0', 'color': COLORS['text_sub']})], md=4, className="border-end px-4", style={'borderColor': COLORS['border']}),
                dbc.Col([html.H5("KONTRA-WSKAŹNIK", className="fw-bold mb-3", style={'color': COLORS['gold']}), html.P(small_desc, style={'fontSize': '1rem', 'marginBottom': '0', 'color': COLORS['text_sub'], 'lineHeight': '1.5'})], md=4, className="border-end px-4", style={'borderColor': COLORS['border']}),
                dbc.Col([html.H5("DYNAMIKA KAPITAŁU", className="fw-bold mb-3", style={'color': COLORS['gold']}), html.P(oi_desc, style={'fontSize': '1rem', 'marginBottom': '0', 'color': COLORS['text_sub'], 'lineHeight': '1.5'})], md=4, className="px-4")
            ])
        ], className="p-4"), style=CARD_STYLE, className="mb-4"), width=12)
    ])

def build_filters_panel():
    return dbc.Row([
        dbc.Col(html.H4("MIKROSTRUKTURA I SEZONOWOŚĆ", className="mb-4 mt-3 fw-bold", style={'color': COLORS['gold'], 'letterSpacing': '1px'}), width=12),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("MIESIĄC", style=LABEL_STYLE, className="mb-2"),
            dbc.Select(id='month-dropdown', options=[{'label': 'CAŁY ROK', 'value': '0'}] + [{'label': m.upper(), 'value': str(i)} for i, m in enumerate(['Styczeń','Luty','Marzec','Kwiecień','Maj','Czerwiec','Lipiec','Sierpień','Wrzesień','Październik','Listopad','Grudzień'], 1)], value='0', style={'backgroundColor': COLORS['bg'], 'color': COLORS['text_main'], 'border': f"1px solid {COLORS['border']}", 'boxShadow': 'none', 'fontSize':'1.1rem', 'padding': '10px'})
        ], className="p-4"), style=CARD_STYLE, className="mb-4"), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("DZIEŃ TRADINGOWY (TDOM)", id='tooltip-tdom-target', style=LABEL_STYLE, className="mb-2"),
            dbc.Tooltip("Trading Day of Month - kolejny dzień roboczy w miesiącu.", target="tooltip-tdom-target", placement="top", style={'fontSize': '1rem'}),
            dbc.Select(id='tdom-dropdown', options=[{'label': f'DZIEŃ {i}', 'value': str(i)} for i in range(1, 24)], value='1', style={'backgroundColor': COLORS['bg'], 'color': COLORS['text_main'], 'border': f"1px solid {COLORS['border']}", 'boxShadow': 'none', 'fontSize':'1.1rem', 'padding': '10px'})
        ], className="p-4"), style=CARD_STYLE, className="mb-4"), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("CENA BAZOWA DO PROJEKCJI (OPEN)", style=LABEL_STYLE, className="mb-2"), 
            dbc.Input(id='today-open-input', value=live_open or 2300.0, type='number', step=0.1, style=dict(MONO, backgroundColor=COLORS['bg'], color=COLORS['text_main'], border=f"1px solid {COLORS['border']}", boxShadow='none', fontSize='1.1rem', padding='10px'))
        ], className="p-4"), style=CARD_STYLE, className="mb-4"), md=4)
    ])

# ==========================================
# 6. GŁÓWNA APLIKACJA
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP])
app.title = "XAU Quant Pro"

app.layout = html.Div([
    dbc.Container([
        build_header(),
        build_cot_panel(),
        build_synthesis_panel(),
        
        dcc.Loading(type="circle", color=COLORS['gold'], children=[
            dbc.Row(id='trading-edge-signals', className="mb-4"),
            
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("INTERMARKET: ZŁOTO VS T-BONDS (10Y)", style=dict(LABEL_STYLE, marginBottom='15px')), dcc.Graph(id='intermarket-chart', config={'displayModeBar': False}, style={'height': '350px'})], className="p-3"), style=CARD_STYLE, className="mb-4"), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("SEZONOWOŚĆ (SKUMULOWANY ZWROT)", style=dict(LABEL_STYLE, marginBottom='15px')), dcc.Graph(id='seasonality-chart', config={'displayModeBar': False}, style={'height': '350px'})], className="p-3"), style=CARD_STYLE, className="mb-4"), md=6)
            ]),
            
            build_filters_panel(),
            
            dbc.Row(id='projection-levels', className="mb-4"),
            html.Div(id='kpi-cards', className="mb-5"),
            
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("WIN RATE (ROZKŁAD HISTORYCZNY)", style=dict(LABEL_STYLE, marginBottom='15px')), dcc.Graph(id='winrate-horizons-chart', config={'displayModeBar': False}, style={'height': '300px'})], className="p-3"), style=CARD_STYLE), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("ZMIENNOŚĆ INTRADAY (UTC)", style=dict(LABEL_STYLE, marginBottom='15px')), dcc.Graph(id='hourly-range-chart', config={'displayModeBar': False}, style={'height': '300px'})], className="p-3"), style=CARD_STYLE), md=6)
            ], className="mb-5")
        ])
    ], fluid=True, className="p-4 px-5")
], style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'})

# ==========================================
# 7. LOGIKA KONTROLERA (CALLBACKS)
# ==========================================
def create_kpi_card(label, value, val_color, span=3, tooltip_text=None, target_id=None):
    label_div = html.Div(label, id=target_id, style=dict(LABEL_STYLE, textAlign='center', marginBottom='10px')) if target_id else html.Div(label, style=dict(LABEL_STYLE, textAlign='center', marginBottom='10px'))
    
    content = [
        label_div,
        html.Div(value, style=dict(MONO, color=val_color, fontSize='2.2rem', textAlign='center', marginTop='10px', fontWeight='bold'))
    ]
    
    if tooltip_text and target_id:
        content.append(dbc.Tooltip(tooltip_text, target=target_id, placement="top", style={'fontSize': '1rem'}))
        
    return dbc.Col(dbc.Card(dbc.CardBody(content, className="p-4"), style=CARD_STYLE, className="mb-4 shadow-sm"), md=span)

@app.callback(
    [Output('kpi-cards', 'children'), Output('projection-levels', 'children'), Output('trading-edge-signals', 'children'),
     Output('cot-bar-chart', 'figure'), Output('intermarket-chart', 'figure'), Output('seasonality-chart', 'figure'),
     Output('winrate-horizons-chart', 'figure'), Output('hourly-range-chart', 'figure')],
    [Input('month-dropdown', 'value'), Input('tdom-dropdown', 'value'), Input('today-open-input', 'value')]
)
def update_dashboard(selected_month_str, selected_tdom_str, open_price):
    if df.empty: return html.Div("Brak danych.", style={'color':COLORS['down'], 'fontSize': '1.2rem'}), [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

    selected_month, selected_tdom = int(selected_month_str), int(selected_tdom_str)
    f_data = df[df['TDOM'] == selected_tdom].copy() if selected_month == 0 else df[(df['Month'] == selected_month) & (df['TDOM'] == selected_tdom)].copy()
    f_clean = f_data.dropna(subset=['MFE_Pct', 'MAE_Pct'])
    
    total_days = len(f_clean)
    month_win_rate, weekday_win_rate = 0.0, 0.0
    m_text, w_text = "SKUT. MIESIĄCA", "SKUT. DNIA TYGODNIA"

    if selected_month != 0:
        months_pl = ['styczeń','luty','marzec','kwiecień','maj','czerwiec','lipiec','sierpień','wrzesień','październik','listopad','grudzień']
        m_text = f"SKUT. MIESIĄCA ({months_pl[selected_month-1].upper()})"
        monthly_profits = df[df['Month'] == selected_month].groupby('Year').apply(lambda x: (x.iloc[-1]['Close'] - x.iloc[0]['Open']) > 0 if not x.empty else False)
        month_win_rate = (monthly_profits.sum() / len(monthly_profits)) * 100 if len(monthly_profits) > 0 else 0
        
        try:
            curr_year = datetime.now().year
            b_days = pd.bdate_range(start=f'{curr_year}-{selected_month:02d}-01', end=f'{curr_year}-{selected_month:02d}-{calendar.monthrange(curr_year, selected_month)[1]}')
            target_weekday_idx = b_days[min(selected_tdom - 1, len(b_days) - 1)].dayofweek
        except Exception: target_weekday_idx = 0 
            
        all_w_data = df[df['Weekday'] == target_weekday_idx].dropna(subset=['Return_Points'])
        weekday_win_rate = (len(all_w_data[all_w_data['Return_Points'] > 0]) / len(all_w_data)) * 100 if not all_w_data.empty else 0
        w_text = f"SKUT. DNIA TYG. ({['pon.', 'wt.', 'śr.', 'czw.', 'pt.', 'sob.', 'nd.'][target_weekday_idx].upper()})"

    open_price = open_price or 0.0
    
    # --- METRYKI QUANT ---
    avg_mfe = f_clean['MFE_Pct'].mean() if total_days > 0 else 0
    avg_mae = f_clean['MAE_Pct'].mean() if total_days > 0 else 0
    max_mfe = f_clean['MFE_Pct'].max() if total_days > 0 else 0
    max_mae = f_clean['MAE_Pct'].max() if total_days > 0 else 0
    tdom_win_rate = (len(f_clean[f_clean['Return_Points'] > 0]) / total_days) * 100 if total_days > 0 else 0
    
    volatility = f_clean['Return_Points'].std() if total_days > 1 else 0
    reward_risk = (avg_mfe / avg_mae) if avg_mae != 0 else 0

    # --- SYGNAŁY GŁÓWNE ---
    c_gold, s_gold = cot_gold.get('Comm_Index', 50), cot_gold.get('Small_Index', 50)
    current_wpr = df['Williams_%R'].iloc[-1] if 'Williams_%R' in df.columns else -50
    entry_signal, sig_color = ("LONG", COLORS['up']) if c_gold >= 75 else (("SHORT", COLORS['down']) if c_gold <= 25 else ("NEUTRAL", COLORS['text_sub']))

    def tip(lbl, col, txt): return html.Li([html.Span(lbl, style={'color': col, 'fontWeight': 'bold', 'marginRight': '10px'}), html.Span(txt, style={'color': COLORS['text_main']})], style={'marginBottom': '10px', 'fontSize': '1.05rem'})
    tips = [
        tip(f"SMART MONEY [{c_gold:.1f}%]:", COLORS['up'] if c_gold >= 75 else (COLORS['down'] if c_gold <= 25 else COLORS['text_sub']), "Skupowanie u dołka." if c_gold >= 75 else ("Dystrybucja na szczycie." if c_gold <= 25 else "Brak przewagi statystycznej.")),
        tip("MOMENTUM (OI):", COLORS['up'] if price_change > 0 and oi_change > 0 else COLORS['down'], "Wzrosty poparte kapitałem." if oi_change > 0 else "Ruch techniczny, bez kapitału.")
    ]
    if entry_signal != "NEUTRAL": tips.append(tip(f"TIMING [%R {current_wpr:.0f}]:", COLORS['gold'], "Optymalny moment wejścia." if (entry_signal=="LONG" and current_wpr<-80) or (entry_signal=="SHORT" and current_wpr>-20) else "Ryzyko lokalnej korekty - zaczekaj."))

    trading_edge_html = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("REKOMENDACJA ZBIORCZA", style=dict(LABEL_STYLE, marginBottom='15px')),
            dbc.Row([
                dbc.Col([html.Div("GLOBAL TREND", style=LABEL_STYLE), html.H2(entry_signal, style={'color': sig_color, 'fontWeight': 'bold', 'letterSpacing': '2px', 'marginTop': '10px', 'fontSize': '2.5rem'})], className="text-center border-end", style={'borderColor': COLORS['border']}),
                dbc.Col([html.Div("TIMING (%R)", style=LABEL_STYLE), html.H2(f"{current_wpr:.0f}", style=dict(MONO, color=COLORS['gold'] if current_wpr < -80 or current_wpr > -20 else COLORS['text_sub'], fontWeight='bold', marginTop='10px', fontSize='2.5rem'))], className="text-center")
            ], className="align-items-center h-100 mt-4")
        ], className="p-4"), style=CARD_STYLE, className="h-100"), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("WNIOSKI SYSTEMOWE", style=dict(LABEL_STYLE, marginBottom='20px')), html.Ul(tips, style={'listStyleType': 'none', 'paddingLeft': '0'})], className="p-4"), style=CARD_STYLE, className="h-100"), md=8)
    ]

    # --- WYKRESY ---
    fig_cot = go.Figure(go.Bar(x=[cot_gold.get('Comm_Net',0), cot_gold.get('NonComm_Net',0), cot_gold.get('Small_Net',0)], y=['COMM', 'LARGE', 'SMALL'], orientation='h', marker_color=[COLORS['up'] if v > 0 else COLORS['down'] for v in [cot_gold.get('Comm_Net',0), cot_gold.get('NonComm_Net',0), cot_gold.get('Small_Net',0)]], text=[format_num(cot_gold.get('Comm_Net',0)), format_num(cot_gold.get('NonComm_Net',0)), format_num(cot_gold.get('Small_Net',0))], textposition='outside', textfont=dict(color=COLORS['text_main'], family="monospace", size=12)))
    fig_cot.update_layout(**BASE_CHART_LAYOUT)
    fig_cot.update_layout(margin=dict(l=0, r=60, t=0, b=0), xaxis=dict(showgrid=False, zeroline=True, zerolinecolor=COLORS['border'], showticklabels=False), yaxis=dict(showgrid=False, tickfont=dict(size=12)))

    fig_im = make_subplots(specs=[[{"secondary_y": True}]])
    if not df_intermarket.empty:
        fig_im.add_trace(go.Scatter(x=df_intermarket.index, y=df_intermarket['Gold'], name="XAU", line=dict(color=COLORS['gold'], width=2)), secondary_y=False)
        fig_im.add_trace(go.Scatter(x=df_intermarket.index, y=df_intermarket['Bonds'], name="10Y BND", line=dict(color=COLORS['blue'], width=2, dash='dot')), secondary_y=True)
    fig_im.update_layout(**BASE_CHART_LAYOUT, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_im.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
    fig_im.update_yaxes(showgrid=True, gridcolor=COLORS['border'], secondary_y=False, tickfont=dict(color=COLORS['gold']))
    fig_im.update_yaxes(showgrid=False, secondary_y=True, tickfont=dict(color=COLORS['blue']))

    fig_seasonality = go.Figure()
    max_year = df['Year'].max() if not df.empty else 2024
    for i, h in enumerate([3, 6, 9, 12, 15, 18, 21]):
        sub_df = df[df['Year'] >= (max_year - h + 1)]
        if not sub_df.empty:
            curve = sub_df.groupby('DayOfYear')['Return_Points'].mean().cumsum()
            fig_seasonality.add_trace(go.Scatter(x=curve.index, y=curve.values, mode='lines', name=f'{h}Y', line=dict(width=2 if h not in [3,12,21] else 3, color=['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#3498db', '#d4af37'][i % 7]), visible=True if h in [3, 12, 21] else 'legendonly'))
    fig_seasonality.update_layout(**BASE_CHART_LAYOUT, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_seasonality.update_xaxes(showgrid=True, gridcolor=COLORS['border'], title="Dzień Roku")
    fig_seasonality.update_yaxes(showgrid=True, gridcolor=COLORS['border'], title="Skumulowany Zysk (Pkt)")

    # --- RENDER KPI ---
    proj_avg_peak, proj_avg_drop = open_price * (1 + (avg_mfe / 100)), open_price * (1 - (avg_mae / 100))
    projections = [
        create_kpi_card("ŚREDNIA PROJEKCJA DZIENNA", f"H: {proj_avg_peak:.1f}  |  L: {proj_avg_drop:.1f}", COLORS['blue'], 6, "Przewidywany zasięg bazujący na historycznej średniej dla tego dnia.", "tooltip-proj-avg"),
        create_kpi_card("PROJEKCJA EKSTREMALNA", f"H: {open_price * (1 + (max_mfe / 100)):.1f}  |  L: {open_price * (1 - (max_mae / 100)):.1f}", COLORS['gold'], 6, "Największe historyczne wychylenie odnotowane w tym dniu.", "tooltip-proj-ext")
    ]

    kpis = html.Div([
        dbc.Row([dbc.Col(html.Div(f"PRÓBKA BADAWCZA: {total_days} SESJI ({f_clean['Year'].nunique() if not f_clean.empty else 0} LAT)", style=dict(LABEL_STYLE, color=COLORS['text_main'], marginBottom='15px', fontSize='1rem')), width=12)]),
        dbc.Row([
            create_kpi_card("WIN RATE (TDOM)", f"{tdom_win_rate:.1f}%", COLORS['text_main'], 3, "Szansa na zamknięcie dnia wyżej niż cena otwarcia.", "t-wr-tdom"),
            create_kpi_card(m_text, f"{month_win_rate:.1f}%", COLORS['neutral'], 3),
            create_kpi_card(w_text, f"{weekday_win_rate:.1f}%", COLORS['neutral'], 3),
            create_kpi_card("REWARD / RISK", f"{reward_risk:.2f}", COLORS['gold'] if reward_risk > 1 else COLORS['down'], 3, "Stosunek średniego maksymalnego zysku do średniego maksymalnego obsunięcia.", "t-rr")
        ]),
        dbc.Row([
            create_kpi_card("AVG UP PULL (MFE)", f"+{avg_mfe:.2f}%", COLORS['up'], 3, "Średnie maksymalne wychylenie ceny w górę w trakcie sesji.", "t-mfe"),
            create_kpi_card("AVG DOWN PULL (MAE)", f"-{avg_mae:.2f}%", COLORS['down'], 3, "Średnie maksymalne obsunięcie ceny w dół w trakcie sesji.", "t-mae"),
            create_kpi_card("MAX UP PULL", f"+{max_mfe:.2f}%", COLORS['up'], 3),
            create_kpi_card("ZMIENNOŚĆ (σ)", f"{volatility:.1f} pkt", COLORS['blue'], 3, "Odchylenie standardowe zwrotów w punktach (historyczna zmienność).", "t-vol")
        ])
    ])

    winrates = []
    for h in [3, 6, 9, 12, 15, 18, 21]:
        sub = f_clean[f_clean['Year'] >= (max_year - h + 1)]
        winrates.append((len(sub[sub['Return_Points'] > 0]) / len(sub) * 100) if len(sub) > 0 else 0)

    fig_wr = go.Figure(go.Bar(x=[f"{h}Y" for h in [3, 6, 9, 12, 15, 18, 21]], y=winrates, marker_color=[COLORS['up'] if w >= 50 else COLORS['down'] for w in winrates]))
    fig_wr.add_hline(y=50, line_dash="dash", line_color=COLORS['border'])
    fig_wr.update_layout(**BASE_CHART_LAYOUT, yaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS['border']))

    if not df_hourly.empty and 'DateOnly' in df_hourly.columns:
        hourly_subset = df_hourly[df_hourly['DateOnly'].isin(f_clean['DateOnly'].unique())]
        if not hourly_subset.empty:
            h_stats = hourly_subset.groupby('Hour')['Return_Points'].mean().reset_index()
            fig_h1 = go.Figure(go.Bar(x=h_stats['Hour'], y=h_stats['Return_Points'], marker_color=[COLORS['up'] if v > 0 else COLORS['down'] for v in h_stats['Return_Points']], text=[f"{v:.1f}" for v in h_stats['Return_Points']], textposition='outside', textfont=dict(color=COLORS['text_main'], family="monospace", size=11)))
            fig_h1.update_layout(**BASE_CHART_LAYOUT, yaxis=dict(showgrid=True, gridcolor=COLORS['border']))
            fig_h1.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        else: fig_h1 = go.Figure().update_layout(**BASE_CHART_LAYOUT)
    else: fig_h1 = go.Figure().update_layout(**BASE_CHART_LAYOUT)

    return kpis, projections, trading_edge_html, fig_cot, fig_im, fig_seasonality, fig_wr, fig_h1

if __name__ == '__main__':
    app.run(debug=True)
