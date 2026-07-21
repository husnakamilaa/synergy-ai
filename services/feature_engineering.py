import numpy as np
import pandas as pd

def minmax_robust(x, invert=False):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    norm = (np.clip(x, lo, hi) - lo) / (hi - lo + 1e-9)
    return (1 - norm) if invert else norm

def fetch_and_calculate_features(conn, id_umkm: str, id_akad_variable: int):
    cursor = conn.cursor()

    # 1. GET data mentah dari akad_variable
    cursor.execute("""
        SELECT aset_lancar, total_hutang_kas, laba_bersih, total_pendapatan, total_beban, aset_tidak_lancar 
        FROM akad_variable 
        WHERE id = %s AND id_umkm = %s
    """, (id_akad_variable, id_umkm))
    var_data = cursor.fetchone()

    if not var_data:
        raise ValueError("Data di tabel akad_variable tidak ditemukan!")

    # 2. GET data historis dari pendapatan_bulanan
    cursor.execute("""
        SELECT id, jumlah, revenue_growth 
        FROM pendapatan_bulanan 
        WHERE id_umkm = %s 
        ORDER BY id ASC
    """, (id_umkm,))
    pend_data = cursor.fetchall()

    if not pend_data or len(pend_data) < 2:
        raise ValueError("Data pendapatan bulanan untuk UMKM ini minimal harus 2 bulan!")

    id_pendapatan_terakhir = pend_data[-1]['id']

    # 3. KALKULASI 6 RASIO
    aset_lancar = float(var_data['aset_lancar'])
    total_hutang_kas = float(var_data['total_hutang_kas'])
    laba_bersih = float(var_data['laba_bersih'])
    total_pendapatan = float(var_data['total_pendapatan'])
    total_beban = float(var_data['total_beban'])
    aset_tidak_lancar = float(var_data['aset_tidak_lancar'])

    incomes = np.array([float(p['jumlah']) for p in pend_data])
    rg_list = [float(p['revenue_growth']) for p in pend_data if p['revenue_growth'] is not None]

    cr = aset_lancar / total_hutang_kas
    npm = laba_bersih / total_pendapatan
    oer = total_beban / total_pendapatan
    cfsr = float(incomes.std(ddof=1) / incomes.mean()) if len(incomes) > 1 else 0.0
    atr = total_pendapatan / (aset_lancar + aset_tidak_lancar)
    rg = float(np.mean(rg_list)) if len(rg_list) > 0 else 0.0

    features_dict = {
        'current_ratio': round(cr, 4),
        'net_profit_margin': round(npm, 4),
        'operating_expense_ratio': round(oer, 4),
        'cashflow_stability_risk': round(cfsr, 4),
        'asset_turnover_ratio': round(atr, 4),
        'revenue_growth': round(rg, 4)
    }

    # 4. HITUNG SKOR KELAYAKAN
    # Menggunakan skala pembobotan rasio
    bobot = {
        'net_profit_margin': 0.25,
        'revenue_growth': 0.20,
        'current_ratio': 0.15,
        'cashflow_stability_risk': 0.15,
        'asset_turnover_ratio': 0.15,
        'operating_expense_ratio': 0.10,
    }
    
    # Estimasi minmax sederhana untuk scoring
    cr_norm = np.clip(cr / 4.0, 0, 1)
    npm_norm = np.clip(npm / 0.5, 0, 1)
    oer_norm = 1.0 - np.clip(oer, 0, 1)
    cfsr_norm = 1.0 - np.clip(cfsr, 0, 1)
    atr_norm = np.clip(atr / 3.5, 0, 1)
    rg_norm = np.clip((rg + 0.5) / 1.0, 0, 1)

    skor01 = (npm_norm * bobot['net_profit_margin'] +
              rg_norm * bobot['revenue_growth'] +
              cr_norm * bobot['current_ratio'] +
              cfsr_norm * bobot['cashflow_stability_risk'] +
              atr_norm * bobot['asset_turnover_ratio'] +
              oer_norm * bobot['operating_expense_ratio'])

    skor_kelayakan = round(float(skor01 * 100), 4)

    cursor.close()
    return features_dict, skor_kelayakan, id_pendapatan_terakhir