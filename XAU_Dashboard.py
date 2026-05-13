import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import dash_bootstrap_components as dbc
import cot_reports as cot
from datetime import datetime
import calendar
import traceback
import io

# ==========================================
# KONFIGURACJA ESTETYKI (INSTITUTIONAL QUANT)
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
    'down': '#ef4444'
}

CARD_STYLE = {
    'backgroundColor': COLORS['panel'],
    'border': f"1px solid {COLORS['border']}",
    'borderRadius': '2px',
    'boxShadow': 'none'
}

MONO = {'fontFamily': "'Consolas', 'Courier New', monospace", 'letterSpacing': '0.5px'}
LABEL_STYLE = {'color': COLORS['text_sub'], 'fontSize': '0.75rem', 'fontWeight': '600', 'letterSpacing': '1px'}

def format_num(val):
    if pd.isna(val): return "0"
    return f"{val:,.0f}".replace(",", " ")

def get_color(val):
    if pd.isna(val) or val == 0: return COLORS['text_main']
    return COLORS['up'] if val > 0 else COLORS['down']

def render_cot_cell(val, chg, is_net=False):
    """Funkcja pomocnicza do renderowania komórek z deltą (zmianą z ubiegłego tygodnia)"""
    if pd.isna(val): val = 0
    if pd.isna(chg): chg = 0
    val_color = get_color(val) if is_net else COLORS['text_main']
    chg_color = COLORS['up'] if chg > 0 else (COLORS['down'] if chg < 0 else COLORS['text_sub'])
    sign = "+" if chg > 0 else ""
    return html.Td([
        html.Div(format_num(val), style={'color': val_color}),
        html.Div(f"{sign}{format_num(chg)}", style={'fontSize': '0.7rem', 'color': chg_color, 'marginTop': '-2px'})
    ], style=dict(MONO, textAlign='right', border='none', verticalAlign='middle', padding='0.5rem 0.5rem'))

# ==========================================
# 1. POBIERANIE DANYCH (MARKET & COT)
# ==========================================
def fetch_live_gold_data(ticker='GC=F', lookback_days=5):
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)
        if df.empty: return None
        df.index = pd.to_datetime(df.index)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return float(df.iloc[-1]['Open'])
    except: return None

def fetch_intermarket_history(lookback='1y'):
    try:
        df_gold = yf.download('GC=F', period=lookback, interval="1d", progress=False)
        if isinstance(df_gold.columns, pd.MultiIndex): df_gold.columns = df_gold.columns.droplevel(1)
        df_bonds = yf.download('ZN=F', period=lookback, interval="1d", progress=False)
        if isinstance(df_bonds.columns, pd.MultiIndex): df_bonds.columns = df_bonds.columns.droplevel(1)
        df = pd.DataFrame({'Gold': df_gold['Close'], 'Bonds': df_bonds['Close']}).dropna()
        return df
    except: return pd.DataFrame()

def fetch_cot_data(commodity_label="GOLD"):
    try:
        current_year = datetime.now().year
        df_cot = cot.cot_year(year=current_year, cot_report_type='legacy_fut')
        if df_cot is None or df_cot.empty: return None, None

        df_cot.columns = df_cot.columns.str.strip().str.replace(' ', '_')
        market_col = next((c for c in df_cot.columns if 'Market' in c and 'Exchange' in c), None)
        if not market_col: market_col = [c for c in df_cot.columns if 'Name' in c][0]

        df_asset = df_cot[df_cot[market_col].astype(str).str.contains(commodity_label, case=False, na=False)].copy()
        if df_asset.empty: return None, None

        date_col = next((c for c in df_asset.columns if 'Date' in c or 'date' in c), None)
        if 'YYMMDD' in date_col:
            df_asset['Parsed_Date'] = pd.to_datetime(df_asset[date_col].astype(str).str.zfill(6), format='%y%m%d', errors='coerce')
        else:
            df_asset['Parsed_Date'] = pd.to_datetime(df_asset[date_col], errors='coerce')
        df_asset = df_asset.dropna(subset=['Parsed_Date']).sort_values('Parsed_Date')

        def get_col(keywords, exclude=[]):
            for col in df_asset.columns:
                cl = col.lower().replace('_', '').replace('-', '')
                if all(k in cl for k in keywords) and not any(e in cl for e in exclude): return col
            return None

        comm_long_col = get_col(['comm', 'long'], exclude=['non'])
        comm_short_col = get_col(['comm', 'short'], exclude=['non'])
        noncomm_long_col = get_col(['noncomm', 'long'])
        noncomm_short_col = get_col(['noncomm', 'short'])
        small_long_col = get_col(['non', 'rep', 'long'], exclude=['comm']) or get_col(['small', 'long'])
        small_short_col = get_col(['non', 'rep', 'short'], exclude=['comm']) or get_col(['small', 'short'])
        oi_col = get_col(['open', 'interest']) or get_col(['oi'])

        def calc_net(long_col, short_col):
            if long_col and short_col: return pd.to_numeric(df_asset[long_col].fillna(0)) - pd.to_numeric(df_asset[short_col].fillna(0))
            return pd.Series([0]*len(df_asset), index=df_asset.index)

        df_asset['Comm_Net'] = calc_net(comm_long_col, comm_short_col)
        df_asset['NonComm_Net'] = calc_net(noncomm_long_col, noncomm_short_col)
        df_asset['Small_Net'] = calc_net(small_long_col, small_short_col)

        window = 26
        def calc_index(series):
            min_val = series.rolling(window, min_periods=1).min()
            max_val = series.rolling(window, min_periods=1).max()
            diff = max_val - min_val
            return np.where(diff == 0, 50, (series - min_val) / diff * 100)

        df_asset['Comm_Index'] = calc_index(df_asset['Comm_Net'])
        df_asset['NonComm_Index'] = calc_index(df_asset['NonComm_Net'])
        df_asset['Small_Index'] = calc_index(df_asset['Small_Net'])

        valid_rows = df_asset.dropna(subset=['Comm_Index'])
        
        # Pobieranie logiki delty (zmiana tydzień do tygodnia)
        if len(valid_rows) >= 2:
            latest = valid_rows.iloc[-1]
            prev = valid_rows.iloc[-2]
        elif not valid_rows.empty:
            latest = valid_rows.iloc[-1]
            prev = latest
        else:
            return None, None

        def safe_get(row, col_name):
            if col_name and col_name in row: return row[col_name]
            return 0

        return {
            'Date': latest['Parsed_Date'].strftime('%Y-%m-%d') if pd.notna(latest['Parsed_Date']) else '---',
            'OI': safe_get(latest, oi_col),
            'OI_Chg': safe_get(latest, oi_col) - safe_get(prev, oi_col),
            
            'Comm_Long': safe_get(latest, comm_long_col), 'Comm_Long_Chg': safe_get(latest, comm_long_col) - safe_get(prev, comm_long_col),
            'Comm_Short': safe_get(latest, comm_short_col), 'Comm_Short_Chg': safe_get(latest, comm_short_col) - safe_get(prev, comm_short_col),
            'Comm_Net': latest['Comm_Net'], 'Comm_Net_Chg': latest['Comm_Net'] - prev['Comm_Net'],
            'Comm_Index': latest['Comm_Index'],
            
            'NonComm_Long': safe_get(latest, noncomm_long_col), 'NonComm_Long_Chg': safe_get(latest, noncomm_long_col) - safe_get(prev, noncomm_long_col),
            'NonComm_Short': safe_get(latest, noncomm_short_col), 'NonComm_Short_Chg': safe_get(latest, noncomm_short_col) - safe_get(prev, noncomm_short_col),
            'NonComm_Net': latest['NonComm_Net'], 'NonComm_Net_Chg': latest['NonComm_Net'] - prev['NonComm_Net'],
            'NonComm_Index': latest['NonComm_Index'],
            
            'Small_Long': safe_get(latest, small_long_col), 'Small_Long_Chg': safe_get(latest, small_long_col) - safe_get(prev, small_long_col),
            'Small_Short': safe_get(latest, small_short_col), 'Small_Short_Chg': safe_get(latest, small_short_col) - safe_get(prev, small_short_col),
            'Small_Net': latest['Small_Net'], 'Small_Net_Chg': latest['Small_Net'] - prev['Small_Net'],
            'Small_Index': latest['Small_Index']
        }, df_asset
    except: return None, None

# ==========================================
# 2. ŁADOWANIE DANYCH HISTORYCZNYCH
# ==========================================
def load_daily_data(filepath='XAU_1d_data.csv'):
    try:
        df = pd.read_csv(filepath, sep=';', index_col=0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        df['Return_Points'] = df['Close'] - df['Open']
        df['MFE_Pct'] = ((df['High'] - df['Open']) / df['Open']) * 100
        df['MAE_Pct'] = ((df['Open'] - df['Low']) / df['Open']) * 100 
        
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['YearMonth'] = df.index.to_period('M')
        df['TDOM'] = df.groupby('YearMonth').cumcount() + 1
        df['Weekday'] = df.index.dayofweek 
        df['DayOfYear'] = df.index.dayofyear
        df['DateOnly'] = df.index.date
        
        df['HH_14'] = df['High'].rolling(window=14).max()
        df['LL_14'] = df['Low'].rolling(window=14).min()
        df['Williams_%R'] = ((df['HH_14'] - df['Close']) / (df['HH_14'] - df['LL_14'])) * -100
        
        return df
    except Exception as e:
        print(f"Błąd ładowania pliku dziennego (XAU_1d_data.csv): {e}")
        return pd.DataFrame()

def load_hourly_data(filepath='XAU_1h_data.csv'):
    try:
        df_h = pd.read_csv(filepath, sep=';')
        df_h['Date'] = pd.to_datetime(df_h['Date'], format='%Y.%m.%d %H:%M')
        df_h.set_index('Date', inplace=True)
        
        df_h['Hour'] = df_h.index.hour
        df_h['Return_Points'] = df_h['Close'] - df_h['Open'] 
        df_h['DateOnly'] = df_h.index.date
        return df_h
    except Exception as e:
        print(f"Błąd ładowania pliku godzinowego (XAU_1h_data.csv): {e}")
        return pd.DataFrame()

# ==========================================
# 3. ZMIENNE GLOBALNE & DYNAMIKA OI
# ==========================================
df = load_daily_data()
df_hourly = load_hourly_data()
df_intermarket = fetch_intermarket_history()
live_open = fetch_live_gold_data()

cot_gold, df_asset_gold = fetch_cot_data("GOLD")
cot_usd, _ = fetch_cot_data("DOLLAR INDEX")

def safe_cot():
    return {
        'Date': '---', 'OI': 0, 'OI_Chg': 0, 
        'Comm_Long': 0, 'Comm_Long_Chg': 0, 'Comm_Short': 0, 'Comm_Short_Chg': 0, 'Comm_Net': 0, 'Comm_Net_Chg': 0, 'Comm_Index': 50, 
        'NonComm_Long': 0, 'NonComm_Long_Chg': 0, 'NonComm_Short': 0, 'NonComm_Short_Chg': 0, 'NonComm_Net': 0, 'NonComm_Net_Chg': 0, 'NonComm_Index': 50, 
        'Small_Long': 0, 'Small_Long_Chg': 0, 'Small_Short': 0, 'Small_Short_Chg': 0, 'Small_Net': 0, 'Small_Net_Chg': 0, 'Small_Index': 50
    }

if cot_gold is None: cot_gold = safe_cot()
if cot_usd is None: cot_usd = safe_cot()

oi_change = 0
price_change = 0
if df_asset_gold is not None and not df_asset_gold.empty and len(df_asset_gold) >= 5:
    oi_col = next((c for c in df_asset_gold.columns if 'open' in c.lower() and 'interest' in c.lower()), None) or next((c for c in df_asset_gold.columns if c.lower() == 'oi'), None)
    if oi_col:
        latest_oi = pd.to_numeric(df_asset_gold.iloc[-1][oi_col], errors='coerce')
        past_oi = pd.to_numeric(df_asset_gold.iloc[-5][oi_col], errors='coerce')
        oi_change = latest_oi - past_oi
        past_date = df_asset_gold.iloc[-5]['Parsed_Date']
        try:
            past_idx = df.index.get_indexer([past_date.replace(tzinfo=None)], method='nearest')[0]
            price_past = df.iloc[past_idx]['Close']
            price_latest = df.iloc[-1]['Close']
            price_change = price_latest - price_past
        except:
            price_change = 0

# --- DYNAMICZNA ANALIZA OPISOWA DANYCH COT ---
c_index = cot_gold.get('Comm_Index', 50)
c_net_chg = cot_gold.get('Comm_Net_Chg', 0)
s_index = cot_gold.get('Small_Index', 50)
oi_chg = cot_gold.get('OI_Chg', 0)

# 1. Smart Money & Delta
if c_index >= 75:
    comm_desc = html.Span([html.B(f"Smart Money silnie kupują (Index: {c_index:.1f}%). ", style={'color': COLORS['up']}), "Jesteśmy w strefie ekstremalnego niedowartościowania. To potężny, historyczny sygnał na zbliżające się WZROSTY."])
elif c_index <= 25:
    comm_desc = html.Span([html.B(f"Smart Money silnie sprzedają (Index: {c_index:.1f}%). ", style={'color': COLORS['down']}), "Jesteśmy w strefie ekstremalnego przewartościowania. Kopalnie szykują się na SPADKI ceny złota."])
else:
    comm_desc = html.Span([html.B(f"Smart Money - Sentyment neutralny (Index: {c_index:.1f}%). ", style={'color': COLORS['text_main']}), "Brak skrajnego ułożenia kapitału. Oczekujemy na wyraźniejszy sygnał kierunkowy."])

if c_net_chg > 0:
    delta_desc = html.Span([html.B("Budowanie Byczej Presji: ", style={'color': COLORS['up']}), f"W ubiegłym tygodniu dołożyli do rynku netto {format_num(c_net_chg)} pozycji długich."])
elif c_net_chg < 0:
    delta_desc = html.Span([html.B("Budowanie Niedźwiedziej Presji: ", style={'color': COLORS['down']}), f"W ubiegłym tygodniu zredukowali rynek netto o {format_num(abs(c_net_chg))} pozycji (agresywne shortowanie)."])
else:
    delta_desc = html.Span("Brak znaczących zmian w pozycjonowaniu w ostatnim tygodniu.", style={'color': COLORS['text_sub']})

# 2. Ulica (Dumb Money)
if s_index >= 75:
    small_desc = html.Span([html.B(f"Ulica w euforii (Index: {s_index:.1f}%). ", style={'color': COLORS['down']}), "Tłum agresywnie kupuje na górce. Złota zasada kontrariańska mówi: jeśli 'ulica' jest pewna wzrostów, szykuj się na brutalną korektę w dół."])
elif s_index <= 25:
    small_desc = html.Span([html.B(f"Ulica w panice (Index: {s_index:.1f}%). ", style={'color': COLORS['up']}), "Drobni inwestorzy uciekają z rynku. Kontrariańsko, to często najlepszy moment, by dołączyć do 'grubych ryb' i szukać pozycji LONG."])
else:
    small_desc = html.Span([html.B(f"Ulica bez kierunku (Index: {s_index:.1f}%). ", style={'color': COLORS['text_main']}), "Drobni spekulanci nie przejawiają skrajnych emocji, wskaźnik znajduje się pośrodku."])

# 3. Open Interest i Momentum
if price_change > 0 and oi_chg > 0:
    oi_desc = html.Span([html.B("ZDROWA HOSSA (Silnik pełen paliwa). ", style={'color': COLORS['up']}), "Cena rośnie przy jednoczesnym wzroście Open Interest. Na rynek wchodzi ogromny, świeży kapitał wspierający długoterminowe wzrosty."])
elif price_change > 0 and oi_chg < 0:
    oi_desc = html.Span([html.B("SHORT SQUEEZE (Paliwo się kończy). ", style={'color': COLORS['down']}), "Cena rośnie, ale OI spada. Wzrosty wynikają głównie z panicznego uciekania sprzedających (zamykanie szortów). Słaby fundament do dalszych wzrostów."])
elif price_change < 0 and oi_chg > 0:
    oi_desc = html.Span([html.B("ZDROWA BESSA (Agresywna dystrybucja). ", style={'color': COLORS['down']}), "Cena spada, a OI drastycznie rośnie. Grubi gracze wlewają na rynek nowy kapitał do grania na spadki."])
elif price_change < 0 and oi_chg < 0:
    oi_desc = html.Span([html.B("KAPITULACJA (Spadki tracą impet). ", style={'color': COLORS['up']}), "Cena i OI spadają. Wyprzedaż ustaje, kapitał powoli ewakuuje się ze zyskownych szortów. Rynek szykuje się do uklepania mocnego dołka."])
else:
    oi_desc = html.Span([html.B("NEUTRALNIE. ", style={'color': COLORS['text_main']}), "Cena i OI nie ukształtowały w ostatnim tygodniu wyraźnej, skorelowanej dynamiki."])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP])
app.title = "XAU | Quant Terminal"

# ==========================================
# 4. LAYOUT APLIKACJI
# ==========================================
app.layout = html.Div([
    dbc.Container([
        
        # HEADER TERMINALA
        dbc.Row([
            dbc.Col(html.H4([
                html.I(className="bi bi-hexagon-fill me-2", style={'color': COLORS['gold']}), 
                "XAU QUANTITATIVE TERMINAL"
            ], className="text-start mt-4 mb-3 font-weight-bold", style={'letterSpacing': '2px', 'color': COLORS['text_main']}), width=12)
        ], className="border-bottom pb-2 mb-4", style={'borderColor': COLORS['border']}),

        # SEKCJA 1: COT
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("RAPORT CFTC (COT) - ZŁOTO", style=LABEL_STYLE),
                        html.Span(f"DATA: {cot_gold['Date']}", style=dict(LABEL_STYLE, float='right'))
                    ], className="mb-3 border-bottom pb-2", style={'borderColor': COLORS['border']}),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("OPEN INTEREST: "),
                                html.Span(format_num(cot_gold['OI'])),
                                html.Span(
                                    f" ({'+' if cot_gold.get('OI_Chg', 0) > 0 else ''}{format_num(cot_gold.get('OI_Chg', 0))})", 
                                    style={
                                        'color': COLORS['up'] if cot_gold.get('OI_Chg', 0) > 0 else (COLORS['down'] if cot_gold.get('OI_Chg', 0) < 0 else COLORS['text_sub']), 
                                        'fontSize': '0.75rem', 
                                        'marginLeft': '5px'
                                    }
                                )
                            ], style=dict(LABEL_STYLE, color=COLORS['gold'], marginBottom='10px')),
                            dbc.Table([
                                html.Thead(html.Tr([
                                    html.Th("GRUPA RYNKOWA", style=dict(LABEL_STYLE, borderBottom=f"1px solid {COLORS['border']}")), 
                                    html.Th("LONG", style=dict(LABEL_STYLE, textAlign='right', borderBottom=f"1px solid {COLORS['border']}")), 
                                    html.Th("SHORT", style=dict(LABEL_STYLE, textAlign='right', borderBottom=f"1px solid {COLORS['border']}")), 
                                    html.Th("NETTO", style=dict(LABEL_STYLE, textAlign='right', borderBottom=f"1px solid {COLORS['border']}")), 
                                    html.Th("INDEX", style=dict(LABEL_STYLE, textAlign='right', borderBottom=f"1px solid {COLORS['border']}"))
                                ])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td("COMMERCIALS", style=dict(LABEL_STYLE, color=COLORS['text_main'], border='none', verticalAlign='middle')), 
                                        render_cot_cell(cot_gold['Comm_Long'], cot_gold['Comm_Long_Chg']),
                                        render_cot_cell(cot_gold['Comm_Short'], cot_gold['Comm_Short_Chg']),
                                        render_cot_cell(cot_gold['Comm_Net'], cot_gold['Comm_Net_Chg'], is_net=True),
                                        html.Td(f"{cot_gold['Comm_Index']:.1f}%", style=dict(MONO, color=COLORS['gold'], textAlign='right', border='none', verticalAlign='middle'))
                                    ]),
                                    html.Tr([
                                        html.Td("LARGE SPECS", style=dict(LABEL_STYLE, color=COLORS['text_main'], border='none', verticalAlign='middle')), 
                                        render_cot_cell(cot_gold['NonComm_Long'], cot_gold['NonComm_Long_Chg']),
                                        render_cot_cell(cot_gold['NonComm_Short'], cot_gold['NonComm_Short_Chg']),
                                        render_cot_cell(cot_gold['NonComm_Net'], cot_gold['NonComm_Net_Chg'], is_net=True),
                                        html.Td(f"{cot_gold['NonComm_Index']:.1f}%", style=dict(MONO, color=COLORS['gold'], textAlign='right', border='none', verticalAlign='middle'))
                                    ]),
                                    html.Tr([
                                        html.Td("SMALL TRADERS", style=dict(LABEL_STYLE, color=COLORS['text_main'], border='none', verticalAlign='middle')), 
                                        render_cot_cell(cot_gold['Small_Long'], cot_gold['Small_Long_Chg']),
                                        render_cot_cell(cot_gold['Small_Short'], cot_gold['Small_Short_Chg']),
                                        render_cot_cell(cot_gold['Small_Net'], cot_gold['Small_Net_Chg'], is_net=True),
                                        html.Td(f"{cot_gold['Small_Index']:.1f}%", style=dict(MONO, color=COLORS['gold'], textAlign='right', border='none', verticalAlign='middle'))
                                    ])
                                ])
                            ], size="sm", className="mb-0 table-borderless"),
                        ], md=7, className="pe-4 border-end", style={'borderColor': COLORS['border']}),
                        
                        dbc.Col([
                            html.Div("POZYCJE NETTO (WIZUALIZACJA)", style=dict(LABEL_STYLE, textAlign='center', marginBottom='0px')),
                            dcc.Graph(id='cot-bar-chart', config={'displayModeBar': False}, style={'height': '140px'})
                        ], md=5, className="ps-4")
                    ])
                ], className="p-4")
            ], style=CARD_STYLE, className="mb-4"), width=12)
        ]),

        # SEKCJA 1.5: BIEŻĄCA ANALIZA OPISOWA
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("BIEŻĄCA ANALIZA DANYCH COT I PALIWA RYNKOWEGO (AI QUANT)", style=LABEL_STYLE),
                    ], className="mb-3 border-bottom pb-2", style={'borderColor': COLORS['border']}),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H6("SMART MONEY & DELTA (COMMERCIALS)", style={'color': COLORS['gold'], 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.P(comm_desc, style={'fontSize': '0.85rem', 'marginBottom': '5px', 'color': COLORS['text_sub']}),
                            html.P(delta_desc, style={'fontSize': '0.85rem', 'marginBottom': '0', 'color': COLORS['text_sub']})
                        ], md=4, className="border-end", style={'borderColor': COLORS['border']}),
                        
                        dbc.Col([
                            html.H6("KONTRA-WSKAŹNIK (SMALL TRADERS)", style={'color': COLORS['gold'], 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.P(small_desc, style={'fontSize': '0.85rem', 'marginBottom': '0', 'color': COLORS['text_sub']})
                        ], md=4, className="border-end", style={'borderColor': COLORS['border']}),
                        
                        dbc.Col([
                            html.H6("DYNAMIKA KAPITAŁU (OPEN INTEREST)", style={'color': COLORS['gold'], 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.P(oi_desc, style={'fontSize': '0.85rem', 'marginBottom': '0', 'color': COLORS['text_sub']})
                        ], md=4)
                    ])
                ], className="p-4")
            ], style=CARD_STYLE, className="mb-4"), width=12)
        ]),

        # SEKCJA 2: TRADING EDGE
        dbc.Row(id='trading-edge-signals', className="mb-4"),

        # SEKCJA 3: WYKRESY (INTERMARKET & SEZONOWOŚĆ)
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div("ANALIZA MIĘDZYRYNKOWA: ZŁOTO VS T-BONDS (10Y)", style=dict(LABEL_STYLE, marginBottom='15px')),
                    dcc.Graph(id='intermarket-chart', config={'displayModeBar': False}, style={'height': '300px'})
                ], className="p-4")
            ], style=CARD_STYLE, className="mb-4"), width=12)
        ]),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div("KRZYWE SEZONOWE (SKUMULOWANY ZWROT % DLA KOLEJNYCH DNI ROKU)", style=dict(LABEL_STYLE, marginBottom='15px')),
                    dcc.Graph(id='seasonality-chart', config={'displayModeBar': False}, style={'height': '400px'})
                ], className="p-4")
            ], style=CARD_STYLE, className="mb-4"), width=12)
        ]),
        
        # SEKCJA 4: FILTRY I BAZA
        dbc.Row([
            dbc.Col(html.H4("ANALIZA STATYSTYCZNA I MIKROSTRUKTURA", className="mb-3 mt-2 font-weight-bold", style={'color': COLORS['gold']}), width=12)
        ]),
        
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("MIESIĄC", style=LABEL_STYLE),
                dbc.Select(
                    id='month-dropdown',
                    options=[{'label': 'CAŁY ROK', 'value': '0'}] + [{'label': m.upper(), 'value': str(i)} for i, m in enumerate(['Styczeń','Luty','Marzec','Kwiecień','Maj','Czerwiec','Lipiec','Sierpień','Wrzesień','Październik','Listopad','Grudzień'], 1)],
                    value='0',
                    style={'backgroundColor': COLORS['bg'], 'color': COLORS['text_main'], 'border': f"1px solid {COLORS['border']}", 'borderRadius': '0px', 'boxShadow': 'none'}
                )
            ]), style=CARD_STYLE, className="mb-4"), md=4),
            
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("DZIEŃ TRADINGOWY (TDOM)", style=LABEL_STYLE),
                dbc.Select(
                    id='tdom-dropdown',
                    options=[{'label': f'DZIEŃ {i}', 'value': str(i)} for i in range(1, 24)],
                    value='1',
                    style={'backgroundColor': COLORS['bg'], 'color': COLORS['text_main'], 'border': f"1px solid {COLORS['border']}", 'borderRadius': '0px', 'boxShadow': 'none'}
                )
            ]), style=CARD_STYLE, className="mb-4"), md=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("CENA BAZOWA DO PROJEKCJI (OPEN)", style=LABEL_STYLE), 
                dbc.Input(id='today-open-input', value=live_open or 2300.00, type='number', step=0.1, 
                          style=dict(MONO, backgroundColor=COLORS['bg'], color=COLORS['text_main'], border=f"1px solid {COLORS['border']}", borderRadius='0px', boxShadow='none'))
            ]), style=CARD_STYLE, className="mb-4"), md=4)
        ]),

        dbc.Row(id='projection-levels', className="mb-4"),
        html.Div(id='kpi-cards', className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("WIN RATE (HORYZONT CZASOWY)", style=dict(LABEL_STYLE, marginBottom='15px')),
                dcc.Graph(id='winrate-horizons-chart', config={'displayModeBar': False}, style={'height': '250px'})
            ]), style=CARD_STYLE), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("PROFIL GODZINOWY (ZMIENNOŚĆ WG CZASU UTC+0)", style=dict(LABEL_STYLE, marginBottom='15px')),
                dcc.Graph(id='hourly-range-chart', config={'displayModeBar': False}, style={'height': '250px'})
            ]), style=CARD_STYLE), md=6)
        ], className="mb-5")

    ], fluid=True, className="p-4")
], style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'})

# ==========================================
# 5. CALLBACKS (SILNIK DASHBOARDU)
# ==========================================
@app.callback(
    [Output('kpi-cards', 'children'),
     Output('projection-levels', 'children'),
     Output('trading-edge-signals', 'children'),
     Output('cot-bar-chart', 'figure'),
     Output('intermarket-chart', 'figure'),
     Output('seasonality-chart', 'figure'),
     Output('winrate-horizons-chart', 'figure'),
     Output('hourly-range-chart', 'figure')],
    [Input('month-dropdown', 'value'),
     Input('tdom-dropdown', 'value'),
     Input('today-open-input', 'value')]
)
def update_all(selected_month_str, selected_tdom_str, open_price):
    
    selected_month = int(selected_month_str)
    selected_tdom = int(selected_tdom_str)

    if df.empty:
        return html.Div("Brak danych dziennych (XAU_1d_data.csv).", style={'color':'red'}), [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

    if selected_month == 0:
        filtered = df[df['TDOM'] == selected_tdom].copy()
    else:
        filtered = df[(df['Month'] == selected_month) & (df['TDOM'] == selected_tdom)].copy()

    weekdays_pl = ['poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela']
    months_pl = ['styczeń','luty','marzec','kwiecień','maj','czerwiec','lipiec','sierpień','wrzesień','październik','listopad','grudzień']
    
    if selected_month != 0:
        month_name = months_pl[selected_month-1]
        month_data = df[df['Month'] == selected_month]
        def is_month_profitable(group):
            if len(group) == 0: return False
            return (group.iloc[-1]['Close'] - group.iloc[0]['Open']) > 0
        monthly_profits = month_data.groupby('Year').apply(is_month_profitable)
        month_win_rate = (monthly_profits.sum() / len(monthly_profits)) * 100 if len(monthly_profits) > 0 else 0
        month_stat_text = f"SKUT. MIESIĄCA ({month_name.upper()})"
        
        current_year = datetime.now().year
        try:
            last_day_of_month = calendar.monthrange(current_year, selected_month)[1]
            b_days = pd.bdate_range(start=f'{current_year}-{selected_month:02d}-01', end=f'{current_year}-{selected_month:02d}-{last_day_of_month}')
            idx = min(selected_tdom - 1, len(b_days) - 1)
            target_weekday_idx = b_days[idx].dayofweek
        except:
            target_weekday_idx = 0 
            
        target_weekday_name = weekdays_pl[target_weekday_idx]
        all_weekday_data = df[df['Weekday'] == target_weekday_idx].dropna(subset=['Return_Points'])
        weekday_win_rate = (len(all_weekday_data[all_weekday_data['Return_Points'] > 0]) / len(all_weekday_data)) * 100 if len(all_weekday_data) > 0 else 0
        weekday_stat_text = f"SKUT. DNIA TYG. ({target_weekday_name.upper()})"
    else:
        month_stat_text = "SKUTECZNOŚĆ MIESIĄCA"
        month_win_rate = 0
        weekday_stat_text = "SKUTECZNOŚĆ DNIA TYGODNIA"
        weekday_win_rate = 0

    if open_price is None: open_price = 0
    filtered_clean = filtered.dropna(subset=['MFE_Pct', 'MAE_Pct'])
    years_in_sample = filtered_clean['Year'].nunique() if not filtered_clean.empty else 0
    total_days = len(filtered_clean)
    
    avg_mfe_pct = filtered_clean['MFE_Pct'].mean() if total_days > 0 else 0
    avg_mae_pct = filtered_clean['MAE_Pct'].mean() if total_days > 0 else 0
    max_mfe_pct = filtered_clean['MFE_Pct'].max() if total_days > 0 else 0
    max_mae_pct = filtered_clean['MAE_Pct'].max() if total_days > 0 else 0
    tdom_win_rate = (len(filtered_clean[filtered_clean['Return_Points'] > 0]) / total_days) * 100 if total_days > 0 else 0

    # 3. Logika Sygnałów Quant
    c_gold = cot_gold.get('Comm_Index', 50)
    c_usd = cot_usd.get('Comm_Index', 50)
    s_gold = cot_gold.get('Small_Index', 50)
    current_wpr = df['Williams_%R'].iloc[-1] if 'Williams_%R' in df.columns else -50
    
    composite_score = abs(c_gold - s_gold) 
    entry_signal = "NEUTRAL"
    signal_color = COLORS['text_sub']
    
    if c_gold >= 75:
        entry_signal = "LONG"
        signal_color = COLORS['up']
    elif c_gold <= 25:
        entry_signal = "SHORT"
        signal_color = COLORS['down']

    tips = []
    def style_tip(label, color, text):
        return html.Li([html.Span(label, style={'color': color, 'fontWeight': 'bold', 'marginRight': '8px'}), html.Span(text, style={'color': COLORS['text_main']})], style={'marginBottom': '8px'})

    if c_gold >= 75: tips.append(style_tip(f"COMMERCIALS [{c_gold:.1f}%]:", COLORS['up'], "Ekstremalne niedowartościowanie. Skupowanie przez Smart Money (Filar Long)."))
    elif c_gold <= 25: tips.append(style_tip(f"COMMERCIALS [{c_gold:.1f}%]:", COLORS['down'], "Ekstremalne przewartościowanie. Dystrybucja przez Smart Money (Filar Short)."))
    else: tips.append(style_tip(f"COMMERCIALS [{c_gold:.1f}%]:", COLORS['text_sub'], "Pozycjonowanie zrównoważone. Brak presji z raportu COT."))

    if entry_signal == "LONG" and s_gold <= 25: tips.append(style_tip(f"ULICA [{s_gold:.1f}%]:", COLORS['gold'], "Idealny układ kontrariański. Tłum panikuje i gra na spadki."))
    elif entry_signal == "SHORT" and s_gold >= 75: tips.append(style_tip(f"ULICA [{s_gold:.1f}%]:", COLORS['gold'], "Idealny układ kontrariański. Tłum kupuje w euforii na górce."))

    if price_change > 0 and oi_change > 0: tips.append(style_tip("PALIWO (OI):", COLORS['up'], "Cena rośnie + OI rośnie. Nowy kapitał wchodzi na rynek, napędzając zdrową hossę."))
    elif price_change > 0 and oi_change < 0: tips.append(style_tip("PALIWO (OI):", COLORS['down'], "Cena rośnie + OI spada. Wzrosty napędzane ucieczką niedźwiedzi. Brak nowego paliwa."))
    elif price_change < 0 and oi_change > 0: tips.append(style_tip("PALIWO (OI):", COLORS['down'], "Cena spada + OI rośnie. Potężna dystrybucja, nowe grube shorty wchodzą na rynek."))
    elif price_change < 0 and oi_change < 0: tips.append(style_tip("PALIWO (OI):", COLORS['blue'], "Cena spada + OI spada. Spadki tracą impet. Szansa na stworzenie się dołka."))

    if entry_signal == "LONG":
        if current_wpr < -80: tips.append(style_tip(f"WYZWALACZ [%R {current_wpr:.0f}]:", COLORS['up'], "Ekstremalne wyprzedanie. Znakomity moment na precyzyjne wejście (LONG)."))
        elif current_wpr > -20: tips.append(style_tip(f"WYZWALACZ [%R {current_wpr:.0f}]:", COLORS['down'], "Wykupienie. ZACZEKAJ na korektę, zanim wejdziesz z prądem fundamentów."))
        else: tips.append(style_tip(f"WYZWALACZ [%R {current_wpr:.0f}]:", COLORS['text_sub'], "Strefa neutralna. Obserwuj akcję cenową na mniejszych interwałach."))
    elif entry_signal == "SHORT":
        if current_wpr > -20: tips.append(style_tip(f"WYZWALACZ [%R {current_wpr:.0f}]:", COLORS['up'], "Ekstremalne wykupienie. Znakomity moment na precyzyjne wejście (SHORT)."))
        elif current_wpr < -80: tips.append(style_tip(f"WYZWALACZ [%R {current_wpr:.0f}]:", COLORS['down'], "Wyprzedanie. ZACZEKAJ na techniczne podbicie, zanim zajmiesz pozycję Short."))
        else: tips.append(style_tip(f"WYZWALACZ [%R {current_wpr:.0f}]:", COLORS['text_sub'], "Strefa neutralna. Obserwuj akcję cenową."))

    trading_edge_html = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("SYSTEM REKOMENDACJI ZBIORCZEJ", style=dict(LABEL_STYLE, marginBottom='15px')),
            dbc.Row([
                dbc.Col([
                    html.Div("GLOBALNY SYGNAŁ", style=dict(LABEL_STYLE, color=COLORS['text_sub'])),
                    html.H2(entry_signal, style={'color': signal_color, 'fontWeight': 'bold', 'letterSpacing': '2px', 'marginTop': '5px'})
                ], className="text-center border-end", style={'borderColor': COLORS['border']}),
                dbc.Col([
                    html.Div("WYZWALACZ (%R)", style=dict(LABEL_STYLE, color=COLORS['text_sub'])),
                    html.H2(f"{current_wpr:.0f}", style=dict(MONO, color=COLORS['gold'] if current_wpr < -80 or current_wpr > -20 else COLORS['text_sub'], fontWeight='bold', marginTop='5px'))
                ], className="text-center")
            ], className="align-items-center h-100 mt-4")
        ]), style=CARD_STYLE, className="h-100"), md=4),
        
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("SYNTEZA WIELOCZYNNIKOWA (WNIOSKI)", style=dict(LABEL_STYLE, marginBottom='15px')),
            html.Ul(tips, style={'listStyleType': 'none', 'paddingLeft': '0', 'fontSize': '0.9rem'})
        ]), style=CARD_STYLE, className="h-100"), md=8)
    ]

    # --- WYKRESY ---
    fig_cot = go.Figure()
    fig_cot.add_trace(go.Bar(
        x=[cot_gold.get('Comm_Net', 0), cot_gold.get('NonComm_Net', 0), cot_gold.get('Small_Net', 0)],
        y=['COMM', 'LARGE', 'SMALL'],
        orientation='h',
        marker_color=[COLORS['down'] if cot_gold.get('Comm_Net', 0) < 0 else COLORS['up'],
                      COLORS['down'] if cot_gold.get('NonComm_Net', 0) < 0 else COLORS['up'],
                      COLORS['down'] if cot_gold.get('Small_Net', 0) < 0 else COLORS['up']],
        text=[format_num(cot_gold.get('Comm_Net', 0)), format_num(cot_gold.get('NonComm_Net', 0)), format_num(cot_gold.get('Small_Net', 0))],
        textposition='outside', textfont=dict(color=COLORS['text_main'], family="monospace")
    ))
    fig_cot.update_layout(
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=40, t=0, b=0), 
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor=COLORS['border'], showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(color=COLORS['text_sub'], size=10, family="monospace"))
    )

    fig_im = make_subplots(specs=[[{"secondary_y": True}]])
    if not df_intermarket.empty:
        fig_im.add_trace(go.Scatter(x=df_intermarket.index, y=df_intermarket['Gold'], name="XAU", line=dict(color=COLORS['gold'], width=1.5)), secondary_y=False)
        fig_im.add_trace(go.Scatter(x=df_intermarket.index, y=df_intermarket['Bonds'], name="10Y BND", line=dict(color=COLORS['blue'], width=1.5, dash='dot')), secondary_y=True)
        fig_im.update_layout(
            template="plotly_dark", plot_bgcolor=COLORS['bg'], paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified", 
            margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLORS['text_sub']))
        )
        fig_im.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
        fig_im.update_yaxes(showgrid=True, gridcolor=COLORS['border'], secondary_y=False, tickfont=dict(color=COLORS['gold'], family="monospace"))
        fig_im.update_yaxes(showgrid=False, secondary_y=True, tickfont=dict(color=COLORS['blue'], family="monospace"))
    else:
        fig_im.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    fig_seasonality = go.Figure()
    season_horizons = [3, 6, 9, 12, 15, 18, 21]
    max_year = df['Year'].max()
    color_seq = ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#3498db', '#d4af37']
    for i, h in enumerate(season_horizons):
        sub_df = df[df['Year'] >= (max_year - h + 1)]
        if not sub_df.empty:
            daily_mean = sub_df.groupby('DayOfYear')['Return_Points'].mean()
            curve = daily_mean.cumsum()
            visible = True if h in [3, 12, 21] else 'legendonly'
            fig_seasonality.add_trace(go.Scatter(x=curve.index, y=curve.values, mode='lines', name=f'{h} Lat', line=dict(width=2, color=color_seq[i % len(color_seq)]), visible=visible))
    fig_seasonality.update_layout(
        template="plotly_dark", plot_bgcolor=COLORS['bg'], paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLORS['text_sub']))
    )
    fig_seasonality.update_xaxes(showgrid=True, gridcolor=COLORS['border'], title="Dzień Roku (TDOY)", title_font=dict(color=COLORS['text_sub']), tickfont=dict(family="monospace", color=COLORS['text_sub']))
    fig_seasonality.update_yaxes(showgrid=True, gridcolor=COLORS['border'], title="Skumulowany Zysk (Pkt)", title_font=dict(color=COLORS['text_sub']), tickfont=dict(family="monospace", color=COLORS['text_sub']))

    # --- KPI I PROJEKCJE ---
    proj_avg_peak = open_price * (1 + (avg_mfe_pct / 100))
    proj_avg_drop = open_price * (1 - (avg_mae_pct / 100))
    proj_max_peak = open_price * (1 + (max_mfe_pct / 100))
    proj_max_drop = open_price * (1 - (max_mae_pct / 100))
    
    projections = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("PROJEKCJA: ŚREDNIA ZMIENNOŚĆ", style=dict(LABEL_STYLE, textAlign='center', color=COLORS['blue'])),
            html.Div(f"H: {proj_avg_peak:.2f} | L: {proj_avg_drop:.2f}", style=dict(MONO, color=COLORS['text_main'], fontSize='1.5rem', textAlign='center', marginTop='10px'))
        ]), style=CARD_STYLE), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("PROJEKCJA: EXTREMA HISTORYCZNE", style=dict(LABEL_STYLE, textAlign='center', color=COLORS['gold'])),
            html.Div(f"H: {proj_max_peak:.2f} | L: {proj_max_drop:.2f}", style=dict(MONO, color=COLORS['text_main'], fontSize='1.5rem', textAlign='center', marginTop='10px'))
        ]), style=CARD_STYLE), md=6)
    ]

    def make_kpi(label, value, val_color):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(label, style=dict(LABEL_STYLE, textAlign='center')),
            html.Div(value, style=dict(MONO, color=val_color, fontSize='1.8rem', textAlign='center', marginTop='10px', fontWeight='bold'))
        ]), style=CARD_STYLE, className="mb-3"), md=3)

    def make_kpi_4(label, value, val_color):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(label, style=dict(LABEL_STYLE, textAlign='center')),
            html.Div(value, style=dict(MONO, color=val_color, fontSize='1.8rem', textAlign='center', marginTop='10px', fontWeight='bold'))
        ]), style=CARD_STYLE), md=4)

    kpis = html.Div([
        dbc.Row([dbc.Col(html.Div(f"DANE: {total_days} SESJI ({years_in_sample} LAT)", style=dict(LABEL_STYLE, color=COLORS['text_main'], marginBottom='15px')), width=12)]),
        dbc.Row([
            make_kpi("SKUTECZNOŚĆ (TDOM)", f"{tdom_win_rate:.1f}%", COLORS['text_main']),
            make_kpi(month_stat_text, f"{month_win_rate:.1f}%", COLORS['blue']),
            make_kpi(weekday_stat_text, f"{weekday_win_rate:.1f}%", COLORS['blue']),
            make_kpi("AVG UP PULL", f"+{avg_mfe_pct:.2f}%", COLORS['up'])
        ]),
        dbc.Row([
            make_kpi_4("AVG DOWN PULL", f"-{avg_mae_pct:.2f}%", COLORS['down']),
            make_kpi_4("MAX UP PULL", f"+{max_mfe_pct:.2f}%", COLORS['up']),
            make_kpi_4("MAX DOWN PULL", f"-{max_mae_pct:.2f}%", COLORS['down'])
        ])
    ])

    horizons, winrates = [3, 6, 9, 12, 15, 18, 21], []
    for h in horizons:
        sub = filtered_clean[filtered_clean['Year'] >= (max_year - h + 1)]
        winrates.append((len(sub[sub['Return_Points'] > 0]) / len(sub) * 100) if len(sub) > 0 else 0)
        
    fig_wr = go.Figure(go.Bar(x=[f"{h}Y" for h in horizons], y=winrates, marker_color=[COLORS['up'] if w >= 50 else COLORS['down'] for w in winrates]))
    fig_wr.add_hline(y=50, line_dash="dash", line_color=COLORS['border'])
    fig_wr.update_layout(
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS['border'], tickfont=dict(family="monospace", color=COLORS['text_sub'])), 
        xaxis=dict(tickfont=dict(family="monospace", color=COLORS['text_sub'])),
        margin=dict(l=0,r=0,t=10,b=0)
    )

    valid_dates = filtered_clean['DateOnly'].unique()
    hourly_subset = df_hourly[df_hourly['DateOnly'].isin(valid_dates)]
    if not hourly_subset.empty:
        hourly_stats = hourly_subset.groupby('Hour')['Return_Points'].mean().reset_index()
        fig_h1 = go.Figure(go.Bar(
            x=hourly_stats['Hour'], y=hourly_stats['Return_Points'], 
            marker_color=[COLORS['up'] if val > 0 else COLORS['down'] for val in hourly_stats['Return_Points']],
            text=[f"{val:.1f}" for val in hourly_stats['Return_Points']], textposition='outside', textfont=dict(color=COLORS['text_main'], family="monospace")
        ))
        fig_h1.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(family="monospace", color=COLORS['text_sub'])),
            xaxis=dict(tickfont=dict(family="monospace", color=COLORS['text_sub']))
        )
        fig_h1.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    else:
        fig_h1 = go.Figure().update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return kpis, projections, trading_edge_html, fig_cot, fig_im, fig_seasonality, fig_wr, fig_h1

if __name__ == '__main__':
    app.run(debug=True)
