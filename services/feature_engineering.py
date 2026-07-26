import numpy as np
import pandas as pd

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
    
    rg_list = []
    
    # Looping mulai dari bulan kedua (index 1), karena bulan pertama gak punya bulan lalu
    for i in range(1, len(pend_data)):
        # 1. Cek isi kolom revenue_growth dari database
        db_rg = pend_data[i]['revenue_growth']
        
        # 2. Logika pengecekan: Ambil jika ada, Hitung jika NULL
        if db_rg is not None:
            # Kalau di DB sudah ada isinya, langsung ambil
            rg_list.append(float(db_rg))
        else:
            # Kalau di DB NULL, kalkulasi manual on-the-fly
            prev_income = incomes[i-1]
            curr_income = incomes[i]
            
            if prev_income > 0:
                growth = (curr_income - prev_income) / prev_income
                rg_list.append(growth)
            else:
                rg_list.append(0.0)

    cr = aset_lancar / total_hutang_kas if total_hutang_kas > 0 else 0.0
    npm = laba_bersih / total_pendapatan if total_pendapatan > 0 else 0.0
    oer = total_beban / total_pendapatan if total_pendapatan > 0 else 0.0
    cfsr = float(incomes.std(ddof=1) / incomes.mean()) if len(incomes) > 1 and incomes.mean() > 0 else 0.0
    atr = total_pendapatan / (aset_lancar + aset_tidak_lancar) if (aset_lancar + aset_tidak_lancar) > 0 else 0.0
    rg = float(np.mean(rg_list)) if len(rg_list) > 0 else 0.0

    features_dict = {
        'current_ratio': round(cr, 4),
        'net_profit_margin': round(npm, 4),
        'operating_expense_ratio': round(oer, 4),
        'cashflow_stability_risk': round(cfsr, 4),
        'asset_turnover_ratio': round(atr, 4),
        'revenue_growth': round(rg, 4)
    }

    # 4. HITUNG SKOR KELAYAKAN (SAW 1-5 & BOBOT ROC)
    # Konversi ke skor 1-5 menggunakan np.select agar cepat
    skor_npm = np.select([npm > 0.20, npm >= 0.12, npm >= 0.05, npm >= 0.01], [5, 4, 3, 2], default=1)
    skor_ato = np.select([atr >= 3.0, atr >= 2.0, atr >= 1.2, atr >= 0.7], [5, 4, 3, 2], default=1)
    skor_cfsr = np.select([cfsr <= 0.25, cfsr <= 0.40, cfsr <= 0.60, cfsr <= 0.85], [5, 4, 3, 2], default=1)
    skor_cr = np.select([cr >= 3.0, cr >= 2.0, cr >= 1.2, cr >= 1.0], [5, 4, 3, 2], default=1)
    
    cond_rg = [(rg >= 0.15) & (rg <= 0.40), (rg >= 0.05) & (rg < 0.15), (rg >= 0.0) & (rg < 0.05), ((rg >= -0.10) & (rg < 0.0)) | (rg > 0.40)]
    skor_rg = np.select(cond_rg, [5, 4, 3, 2], default=1)
    
    skor_oer = np.select([oer < 0.55, oer <= 0.70, oer <= 0.85, oer <= 0.95], [5, 4, 3, 2], default=1)

    # Normalisasi (S - 1) / 4
    r_npm, r_ato, r_cfsr = (skor_npm - 1)/4, (skor_ato - 1)/4, (skor_cfsr - 1)/4
    r_cr, r_rg, r_oer    = (skor_cr - 1)/4, (skor_rg - 1)/4, (skor_oer - 1)/4

    # Agregasi Bobot ROC
    skor01 = (r_npm * 0.4083 + r_ato * 0.2417 + r_cfsr * 0.1583 + 
              r_cr * 0.1028 + r_rg * 0.0611 + r_oer * 0.0278)

    skor_kelayakan = round(float(skor01 * 100), 4)

    cursor.close()
    return features_dict, skor_kelayakan, id_pendapatan_terakhir