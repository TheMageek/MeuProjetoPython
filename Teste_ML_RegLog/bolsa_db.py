import warnings
warnings.filterwarnings("ignore")

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
from tqdm import tqdm
from datetime import datetime, date

DB_PATH    = "bolsa_b3.db"
START_DATE = "2018-01-01"
END_DATE   = date.today().strftime("%Y-%m-%d")

ACOES_POR_SETOR = {
    "Petróleo e Gás": [
        "PETR3.SA", "PETR4.SA", "RECV3.SA", "PRIO3.SA",
        "RRRP3.SA", "UGPA3.SA", "VBBR3.SA", "CSAN3.SA",
    ],
    "Mineração e Siderurgia": [
        "VALE3.SA", "CSNA3.SA", "GGBR4.SA", "USIM5.SA",
        "GOAU4.SA", "FESA4.SA", "CMIN3.SA", "BRAP4.SA",
    ],
    "Financeiro e Bancos": [
        "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA",
        "BRSR6.SA", "BMGB4.SA", "BPAC11.SA", "AIAB3.SA",
        "IRBR3.SA", "SULA11.SA", "PSSA3.SA", "BBSE3.SA",
    ],
    "Energia Elétrica": [
        "ELET3.SA", "ELET6.SA", "ENEV3.SA", "CPFE3.SA",
        "CMIG4.SA", "EGIE3.SA", "ENGI11.SA", "TAEE11.SA",
        "AURE3.SA", "CPLE6.SA", "TRPL4.SA", "EQTL3.SA",
    ],
    "Varejo e Consumo": [
        "MGLU3.SA", "VIIA3.SA", "AMER3.SA", "LREN3.SA",
        "ALPA4.SA", "SOMA3.SA", "CEAB3.SA", "AMAR3.SA",
        "PETZ3.SA", "SBFG3.SA", "VIVA3.SA", "GRND3.SA",
    ],
    "Alimentos e Bebidas": [
        "ABEV3.SA", "JBSS3.SA", "BRFS3.SA", "MRFG3.SA",
        "BEEF3.SA", "SMFT3.SA", "MDIA3.SA", "CAML3.SA",
        "SLCE3.SA", "TTEN3.SA",
    ],
    "Saúde e Farmácia": [
        "RDOR3.SA", "HAPV3.SA", "GNDI3.SA", "FLRY3.SA",
        "DASA3.SA", "RADL3.SA", "PNVL3.SA", "ONCO3.SA",
        "AALR3.SA", "BLAU3.SA", "HYPE3.SA",
    ],
    "Construção e Imóveis": [
        "MRVE3.SA", "CYRE3.SA", "EZTC3.SA", "TEND3.SA",
        "EVEN3.SA", "DIRR3.SA", "PLPL3.SA", "JHSF3.SA",
        "HBOR3.SA", "LAVV3.SA", "TRIS3.SA",
    ],
    "Telecomunicações": [
        "VIVT3.SA", "TIMS3.SA", "OIBR3.SA", "OIBR4.SA",
        "FIQE3.SA",
    ],
    "Tecnologia": [
        "TOTVS3.SA", "INTB3.SA", "LWSA3.SA", "POSI3.SA",
        "SQIA3.SA", "CASH3.SA", "BMOB3.SA", "IFCM3.SA",
    ],
    "Logística e Transporte": [
        "RAIL3.SA", "CCRO3.SA", "ECOR3.SA", "GOLL4.SA",
        "AZUL4.SA", "TGMA3.SA", "HBSA3.SA", "VAMO3.SA",
    ],
    "Agronegócio": [
        "AGRO3.SA", "TTEN3.SA", "SMTO3.SA", "SOJA3.SA",
        "LAND3.SA", "TASA4.SA", "FIQE3.SA",
    ],
    "Papel e Celulose": [
        "SUZB3.SA", "KLBN11.SA", "RANI3.SA",
    ],
    "Saneamento": [
        "SAPR11.SA", "CSMG3.SA", "SBSP3.SA", "AESB3.SA",
    ],
    "Shopping e Imóveis Comerciais": [
        "MULT3.SA", "IGTI11.SA", "BRML3.SA", "ALLOS3.SA",
        "HMLA11.SA",
    ],
}

# Macros relevantes por setor
MACRO_POR_SETOR = {
    "Petróleo e Gás":              ["brent", "dolar", "ibov", "vix"],
    "Mineração e Siderurgia":      ["minerio", "dolar", "ibov", "cobre"],
    "Financeiro e Bancos":         ["dolar", "ibov", "sp500", "vix"],
    "Energia Elétrica":            ["ibov", "dolar", "igpm"],
    "Varejo e Consumo":            ["ibov", "dolar", "ipca"],
    "Alimentos e Bebidas":         ["dolar", "ibov", "soja", "milho"],
    "Saúde e Farmácia":            ["ibov", "dolar", "ipca"],
    "Construção e Imóveis":        ["ibov", "dolar", "igpm", "ipca"],
    "Telecomunicações":            ["ibov", "dolar", "vix"],
    "Tecnologia":                  ["ibov", "sp500", "dolar", "vix"],
    "Logística e Transporte":      ["ibov", "dolar", "brent"],
    "Agronegócio":                 ["dolar", "soja", "milho", "ibov"],
    "Papel e Celulose":            ["dolar", "ibov", "celulose"],
    "Saneamento":                  ["ibov", "igpm", "dolar"],
    "Shopping e Imóveis Comerciais": ["ibov", "igpm", "ipca", "dolar"],
}

MACRO_TICKERS_GLOBAL = {
    # já existentes
    "brent"   : "BZ=F",
    "dolar"   : "USDBRL=X",
    "ibov"    : "^BVSP",
    "sp500"   : "^GSPC",
    "vix"     : "^VIX",
    "ouro"    : "GC=F",
    "minerio" : "TIO=F",
    "cobre"   : "HG=F",
    "soja"    : "ZS=F",
    "milho"   : "ZC=F",
    "gasoline": "RB=F",      # Gasolina (RBOB)
    "natgas"  : "NG=F",      # Gás natural
    "aco"     : "SBR00.L",   # Aço (London Metal Exchange)
    "diesel"  : "HO=F",      # Heating oil — proxy de diesel
    "bdi"     : "^BDI",      # Baltic Dry Index
    "acucar"  : "SB=F",      # Açúcar bruto
    "cafe"    : "KC=F",      # Café arábica
    "trigo"   : "ZW=F",      # Trigo
    "algodao" : "CT=F",      # Algodão
    "t2y"     : "^IRX",      # Treasury 2 anos (juro curto EUA)
    "t10y"    : "^TNX",      # Treasury 10 anos (juro longo EUA)
    "euro"    : "EURUSD=X",  # Euro/dólar
    "yuan"    : "CNYBRL=X",  # Yuan/real
    "nasdaq"  : "^IXIC",     # Nasdaq Composite
    "xlv"     : "XLV",       # ETF saúde EUA
    "xlre"    : "XLRE",      # ETF real estate EUA
}


SCHEMA = """
-- Tabela de ações (cadastro)
CREATE TABLE IF NOT EXISTS acoes (
    ticker      TEXT PRIMARY KEY,
    setor       TEXT NOT NULL,
    nome        TEXT,
    ativo       INTEGER DEFAULT 1,
    criado_em   TEXT DEFAULT (datetime('now'))
);

-- Preços históricos diários (OHLCV)
CREATE TABLE IF NOT EXISTS precos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    data        TEXT NOT NULL,
    abertura    REAL,
    maxima      REAL,
    minima      REAL,
    fechamento  REAL,
    volume      REAL,
    UNIQUE(ticker, data),
    FOREIGN KEY (ticker) REFERENCES acoes(ticker)
);

-- Indicadores técnicos diários
CREATE TABLE IF NOT EXISTS indicadores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    data          TEXT NOT NULL,
    rsi_14        REAL,
    macd          REAL,
    macd_signal   REAL,
    macd_hist     REAL,
    bb_pct        REAL,
    bb_width      REAL,
    sma_5         REAL,
    sma_20        REAL,
    sma_50        REAL,
    vol_20        REAL,
    volume_ratio  REAL,
    momentum_10   REAL,
    UNIQUE(ticker, data),
    FOREIGN KEY (ticker) REFERENCES acoes(ticker)
);

-- Dados fundamentalistas (snapshot mais recente)
CREATE TABLE IF NOT EXISTS fundamentalistas (
    ticker              TEXT PRIMARY KEY,
    data_atualizacao    TEXT,
    pl                  REAL,   -- Preço/Lucro
    pvp                 REAL,   -- Preço/Valor Patrimonial
    dy                  REAL,   -- Dividend Yield %
    roe                 REAL,   -- Return on Equity %
    margem_liquida      REAL,
    divida_pl           REAL,   -- Dívida / PL
    market_cap          REAL,
    beta                REAL,
    FOREIGN KEY (ticker) REFERENCES acoes(ticker)
);

-- Dados macroeconômicos diários
CREATE TABLE IF NOT EXISTS macro (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    data    TEXT NOT NULL,
    ativo   TEXT NOT NULL,
    valor   REAL,
    retorno REAL,
    UNIQUE(data, ativo)
);

-- Log de atualizações
CREATE TABLE IF NOT EXISTS log_atualizacoes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT,
    tipo        TEXT,
    status      TEXT,
    mensagem    TEXT,
    data_hora   TEXT DEFAULT (datetime('now'))
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_precos_ticker_data   ON precos(ticker, data);
CREATE INDEX IF NOT EXISTS idx_indicadores_ticker   ON indicadores(ticker, data);
CREATE INDEX IF NOT EXISTS idx_macro_data           ON macro(data, ativo);
"""

def criar_banco():
    print(f"[DB] Criando banco de dados: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"[DB] Schema criado com sucesso.")


def cadastrar_acoes():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    total = 0
    for setor, tickers in ACOES_POR_SETOR.items():
        for ticker in tickers:
            cur.execute("""
                INSERT OR IGNORE INTO acoes (ticker, setor)
                VALUES (?, ?)
            """, (ticker, setor))
            total += cur.rowcount
    conn.commit()
    conn.close()
    print(f"[CADASTRO] {total} ações cadastradas em {len(ACOES_POR_SETOR)} setores.")



def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    ind = pd.DataFrame(index=df.index)


    ind["rsi_14"] = compute_rsi(close, 14)


    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"]        = ema12 - ema26
    ind["macd_signal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macd_hist"]   = ind["macd"] - ind["macd_signal"]


    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_up  = sma20 + 2 * std20
    bb_lo  = sma20 - 2 * std20
    ind["bb_pct"]   = (close - bb_lo) / (bb_up - bb_lo + 1e-10)
    ind["bb_width"] = (bb_up - bb_lo) / (sma20 + 1e-10)

    ind["sma_5"]  = close.rolling(5).mean()
    ind["sma_20"] = sma20
    ind["sma_50"] = close.rolling(50).mean()

    ret = close.pct_change()
    ind["vol_20"] = ret.rolling(20).std()

    ind["volume_ratio"] = vol / (vol.rolling(20).mean() + 1e-10)

    ind["momentum_10"] = close / (close.shift(10) + 1e-10) - 1

    return ind.round(6)


def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def salvar_precos(conn, ticker, df_price):
    rows = []
    for dt, row in df_price.iterrows():
        rows.append((
            ticker,
            str(dt.date()),
            round(float(row["Open"]),   4) if pd.notna(row["Open"])   else None,
            round(float(row["High"]),   4) if pd.notna(row["High"])   else None,
            round(float(row["Low"]),    4) if pd.notna(row["Low"])    else None,
            round(float(row["Close"]),  4) if pd.notna(row["Close"])  else None,
            round(float(row["Volume"]), 2) if pd.notna(row["Volume"]) else None,
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO precos
            (ticker, data, abertura, maxima, minima, fechamento, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)

def salvar_indicadores(conn, ticker, ind_df):
    rows = []
    for dt, row in ind_df.iterrows():
        def v(x):
            return float(x) if pd.notna(x) else None
        rows.append((
            ticker, str(dt.date()),
            v(row.get("rsi_14")),   v(row.get("macd")),
            v(row.get("macd_signal")), v(row.get("macd_hist")),
            v(row.get("bb_pct")),   v(row.get("bb_width")),
            v(row.get("sma_5")),    v(row.get("sma_20")),
            v(row.get("sma_50")),   v(row.get("vol_20")),
            v(row.get("volume_ratio")), v(row.get("momentum_10")),
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO indicadores
            (ticker, data, rsi_14, macd, macd_signal, macd_hist,
             bb_pct, bb_width, sma_5, sma_20, sma_50,
             vol_20, volume_ratio, momentum_10)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

def download_precos_todos(start=START_DATE, end=END_DATE):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT ticker FROM acoes WHERE ativo = 1")
    tickers = [r[0] for r in cur.fetchall()]

    print(f"\n[PREÇOS] Baixando {len(tickers)} ações ({start} → {end})...")
    ok, erro = 0, 0

    for ticker in tqdm(tickers, desc="Ações"):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < 10:
                raise ValueError("Dados insuficientes")

            df = flatten_cols(df)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            salvar_precos(conn, ticker, df)
            ind = compute_indicators(df)
            salvar_indicadores(conn, ticker, ind)

            conn.execute("""
                INSERT INTO log_atualizacoes (ticker, tipo, status)
                VALUES (?, 'precos', 'ok')
            """, (ticker,))
            ok += 1

        except Exception as e:
            conn.execute("""
                INSERT INTO log_atualizacoes (ticker, tipo, status, mensagem)
                VALUES (?, 'precos', 'erro', ?)
            """, (ticker, str(e)))
            erro += 1

        conn.commit()
        time.sleep(0.3)  

    conn.close()
    print(f"[PREÇOS] Concluído: {ok} OK | {erro} erros")

def download_fundamentalistas():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT ticker FROM acoes WHERE ativo = 1")
    tickers = [r[0] for r in cur.fetchall()]

    print(f"\n[FUND.] Baixando fundamentalistas de {len(tickers)} ações...")
    ok, erro = 0, 0

    for ticker in tqdm(tickers, desc="Fundamentalistas"):
        try:
            info = yf.Ticker(ticker).info
            if not info or "symbol" not in info:
                raise ValueError("Sem dados de info")

           
            nome = info.get("longName") or info.get("shortName", "")
            conn.execute("UPDATE acoes SET nome = ? WHERE ticker = ?",
                         (nome, ticker))

            def safe(key, divisor=1):
                val = info.get(key)
                if val is None or val == "Infinity":
                    return None
                try:
                    return round(float(val) / divisor, 4)
                except:
                    return None

            conn.execute("""
                INSERT OR REPLACE INTO fundamentalistas
                    (ticker, data_atualizacao, pl, pvp, dy, roe,
                     margem_liquida, divida_pl, market_cap, beta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                date.today().strftime("%Y-%m-%d"),
                safe("trailingPE"),
                safe("priceToBook"),
                safe("dividendYield", 0.01),   
                safe("returnOnEquity", 0.01),
                safe("profitMargins", 0.01),
                safe("debtToEquity"),
                safe("marketCap"),
                safe("beta"),
            ))

            conn.execute("""
                INSERT INTO log_atualizacoes (ticker, tipo, status)
                VALUES (?, 'fundamentalistas', 'ok')
            """, (ticker,))
            ok += 1

        except Exception as e:
            conn.execute("""
                INSERT INTO log_atualizacoes (ticker, tipo, status, mensagem)
                VALUES (?, 'fundamentalistas', 'erro', ?)
            """, (ticker, str(e)))
            erro += 1

        conn.commit()
        time.sleep(0.5)

    conn.close()
    print(f"[FUND.] Concluído: {ok} OK | {erro} erros")


def download_macro(start=START_DATE, end=END_DATE):
    print(f"\n[MACRO] Baixando {len(MACRO_TICKERS_GLOBAL)} ativos macro...")
    conn = sqlite3.connect(DB_PATH)
    ok, erro = 0, 0

    for nome, sym in tqdm(MACRO_TICKERS_GLOBAL.items(), desc="Macro"):
        try:
            df = yf.download(sym, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("Sem dados")
            df = flatten_cols(df)
            close = df["Close"].squeeze()
            ret   = close.pct_change()

            rows = []
            for dt in close.index:
                v = close.loc[dt]
                r = ret.loc[dt]
                rows.append((
                    str(dt.date()), nome,
                    round(float(v), 6) if pd.notna(v) else None,
                    round(float(r), 6) if pd.notna(r) else None,
                ))
            conn.executemany("""
                INSERT OR REPLACE INTO macro (data, ativo, valor, retorno)
                VALUES (?, ?, ?, ?)
            """, rows)
            conn.commit()
            ok += 1

        except Exception as e:
            conn.execute("""
                INSERT INTO log_atualizacoes (ticker, tipo, status, mensagem)
                VALUES (?, 'macro', 'erro', ?)
            """, (nome, str(e)))
            conn.commit()
            erro += 1

        time.sleep(0.3)

    conn.close()
    print(f"[MACRO] Concluído: {ok} OK | {erro} erros")



def get_precos(ticker, n_dias=252):
    """Retorna os últimos N dias de preços de uma ação."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT data, abertura, maxima, minima, fechamento, volume
        FROM precos
        WHERE ticker = ?
        ORDER BY data DESC
        LIMIT ?
    """, conn, params=(ticker, n_dias))
    conn.close()
    df["data"] = pd.to_datetime(df["data"])
    return df.set_index("data").sort_index()

def get_indicadores(ticker, n_dias=252):
    """Retorna os últimos N dias de indicadores de uma ação."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM indicadores
        WHERE ticker = ?
        ORDER BY data DESC
        LIMIT ?
    """, conn, params=(ticker, n_dias))
    conn.close()
    df["data"] = pd.to_datetime(df["data"])
    return df.set_index("data").sort_index()

def get_fundamentalistas(ticker):
    """Retorna dados fundamentalistas de uma ação."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT f.*, a.setor, a.nome
        FROM fundamentalistas f
        JOIN acoes a ON f.ticker = a.ticker
        WHERE f.ticker = ?
    """, conn, params=(ticker,))
    conn.close()
    return df

def get_setor(setor):
    """Retorna todas as ações de um setor com seus dados fundamentalistas."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT a.ticker, a.nome, a.setor,
               f.pl, f.pvp, f.dy, f.roe, f.market_cap, f.beta
        FROM acoes a
        LEFT JOIN fundamentalistas f ON a.ticker = f.ticker
        WHERE a.setor = ?
        ORDER BY f.market_cap DESC
    """, conn, params=(setor,))
    conn.close()
    return df

def get_macro_para_setor(setor, n_dias=252):
    """Retorna dados macro relevantes para um setor específico."""
    ativos = MACRO_POR_SETOR.get(setor, ["ibov", "dolar"])
    conn   = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(ativos))
    df = pd.read_sql_query(f"""
        SELECT data, ativo, valor, retorno
        FROM macro
        WHERE ativo IN ({placeholders})
        ORDER BY data DESC
        LIMIT ?
    """, conn, params=ativos + [n_dias * len(ativos)])
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["data"] = pd.to_datetime(df["data"])
    pivot = df.pivot(index="data", columns="ativo", values="retorno").sort_index()
    return pivot

def listar_setores():
    """Mostra quantas ações há por setor."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT setor, COUNT(*) as total_acoes
        FROM acoes
        WHERE ativo = 1
        GROUP BY setor
        ORDER BY total_acoes DESC
    """, conn)
    conn.close()
    return df

def resumo_banco():
    """Mostra estatísticas gerais do banco."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    stats = {}
    for tabela in ["acoes", "precos", "indicadores", "fundamentalistas", "macro"]:
        cur.execute(f"SELECT COUNT(*) FROM {tabela}")
        stats[tabela] = cur.fetchone()[0]

    cur.execute("SELECT MIN(data), MAX(data) FROM precos")
    datas = cur.fetchone()
    conn.close()

    print("\n" + "=" * 46)
    print("  RESUMO DO BANCO DE DADOS")
    print("=" * 46)
    for tabela, qtd in stats.items():
        print(f"  {tabela:<20}: {qtd:>8,} registros")
    if datas[0]:
        print(f"  Período de preços    : {datas[0]} → {datas[1]}")
    print("=" * 46)



def preparar_dados_para_modelo(ticker, n_dias=500):
    """
    Retorna um DataFrame com preços + indicadores + macro do setor,
    pronto para usar no stock_predictor_v2.py
    """
    conn = sqlite3.connect(DB_PATH)

    df_p = pd.read_sql_query("""
        SELECT data, abertura as Open, maxima as High, minima as Low,
               fechamento as Close, volume as Volume
        FROM precos WHERE ticker = ?
        ORDER BY data DESC LIMIT ?
    """, conn, params=(ticker, n_dias))

    df_i = pd.read_sql_query("""
        SELECT * FROM indicadores WHERE ticker = ?
        ORDER BY data DESC LIMIT ?
    """, conn, params=(ticker, n_dias))

    conn.close()

    if df_p.empty:
        raise ValueError(f"Sem dados de preço para {ticker}. Rode o download primeiro.")

    df_p["data"] = pd.to_datetime(df_p["data"])
    df_i["data"] = pd.to_datetime(df_i["data"])
    df_p = df_p.set_index("data").sort_index()
    df_i = df_i.set_index("data").sort_index().drop(
        columns=["id", "ticker"], errors="ignore")

    df = df_p.join(df_i, how="left")

    conn2 = sqlite3.connect(DB_PATH)
    row = conn2.execute(
        "SELECT setor FROM acoes WHERE ticker = ?", (ticker,)
    ).fetchone()
    conn2.close()

    if row:
        setor = row[0]
        macro = get_macro_para_setor(setor, n_dias)
        if not macro.empty:
            df = df.join(macro, how="left")

    df.ffill(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    return df



if __name__ == "__main__":
    print("=" * 54)
    print("  B3 Database Builder")
    print("=" * 54)

    criar_banco()

    cadastrar_acoes()

    print("\nSetores cadastrados:")
    for setor, tickers in ACOES_POR_SETOR.items():
        print(f"  {setor:<35} {len(tickers)} ações")

    print("\nO que deseja executar?")
    print("  1 - Tudo (preços + fundamentalistas + macro)")
    print("  2 - Só preços e indicadores")
    print("  3 - Só fundamentalistas")
    print("  4 - Só macro")
    print("  5 - Só ver resumo do banco (se já populado)")
    print("  6 - Testar consulta (ex: PETR4.SA)")

    escolha = input("\nEscolha [1-6]: ").strip()

    if escolha == "1":
        download_precos_todos()
        download_fundamentalistas()
        download_macro()
        resumo_banco()

    elif escolha == "2":
        download_precos_todos()
        resumo_banco()

    elif escolha == "3":
        download_fundamentalistas()
        resumo_banco()

    elif escolha == "4":
        download_macro()
        resumo_banco()

    elif escolha == "5":
        resumo_banco()
        print("\nAções por setor:")
        print(listar_setores().to_string(index=False))

    elif escolha == "6":
        ticker = input("Ticker (ex: PETR4.SA): ").strip().upper()
        if not ticker.endswith(".SA"):
            ticker += ".SA"
        print(f"\nÚltimos 5 preços de {ticker}:")
        print(get_precos(ticker, 5))
        print(f"\nFundamentalistas de {ticker}:")
        print(get_fundamentalistas(ticker))
        print(f"\nMacro do setor:")
        conn_tmp = sqlite3.connect(DB_PATH)
        setor = conn_tmp.execute(
            "SELECT setor FROM acoes WHERE ticker = ?", (ticker,)
        ).fetchone()
        conn_tmp.close()
        if setor:
            print(get_macro_para_setor(setor[0], 5))

    print("\nPronto! Banco salvo em:", os.path.abspath(DB_PATH))