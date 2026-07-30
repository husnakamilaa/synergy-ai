import numpy as np
import pandas as pd

def fetch_and_calculate_features(conn, id_umkm: str, id_akad_variable: int):
    cursor = conn.cursor()

    # GET : data variable akad
    cursor.execute("""
        SELECT aset_lancar, total_hutang_kas, laba_bersih, total_pendapatan, total_beban, aset_tidak_lancar 
        FROM akad_variable 
        WHERE id = %s AND id_umkm = %s
    """, (id_akad_variable, id_umkm))
    var_data = cursor.fetchone()

    if not var_data:
        raise ValueError("Data di tabel akad_variable tidak ditemukan!")

    # GET : pendapatan bulanan
    cursor.execute("""
        SELECT id, bulan, tahun, jumlah, revenue_growth 
        FROM pendapatan_bulanan 
        WHERE id_umkm = %s 
        ORDER BY id ASC
    """, (id_umkm,))
    pend_data = cursor.fetchall()

    if not pend_data or len(pend_data) < 2:
        raise ValueError("Data pendapatan bulanan untuk UMKM ini minimal harus 2 bulan!")

    # 1. BIKIN KAMUS BULAN
    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    # Masukkan angka bulan ke dalam tiap baris data
    for p in pend_data:
        nama_bulan = p['bulan'].strip().capitalize()
        p['bulan_angka'] = bulan_map.get(nama_bulan, 0)
        p['tahun'] = int(p['tahun'])
        p['jumlah'] = float(p['jumlah'])

    # 2. URUTKAN DATA SECARA KRONOLOGIS (Tahun dulu, baru Bulan)
    pend_data_sorted = sorted(pend_data, key=lambda x: (x['tahun'], x['bulan_angka']))
    id_pendapatan_terakhir = pend_data_sorted[-1]['id']

    # KALKULASI RASIO
    aset_lancar = float(var_data['aset_lancar'])
    total_hutang_kas = float(var_data['total_hutang_kas'])
    laba_bersih = float(var_data['laba_bersih'])
    total_pendapatan = float(var_data['total_pendapatan'])
    total_beban = float(var_data['total_beban'])
    aset_tidak_lancar = float(var_data['aset_tidak_lancar'])

    # Pastikan array incomes menggunakan data yang sudah urut (penting untuk perhitungan CFSR)
    incomes = np.array([float(p['jumlah']) for p in pend_data_sorted])
    
    rg_list = []

    # 3. LOOPING HITUNG GROWTH MENGGUNAKAN CMGR
    for i in range(1, len(pend_data_sorted)):
        curr_data = pend_data_sorted[i]
        prev_data = pend_data_sorted[i-1]
        
        db_rg = curr_data['revenue_growth']
        
        if db_rg is not None:
            rg_list.append(float(db_rg))
        else:
            prev_income = prev_data['jumlah']
            curr_income = curr_data['jumlah']
            
            # Hitung jarak bulan (n)
            gap_tahun = curr_data['tahun'] - prev_data['tahun']
            gap_bulan = curr_data['bulan_angka'] - prev_data['bulan_angka']
            n_months = (gap_tahun * 12) + gap_bulan
            
            if prev_income > 0 and n_months > 0:
                growth = (curr_income / prev_income) ** (1 / n_months) - 1
                rg_list.append(growth)
            elif prev_income > 0 and n_months <= 0:
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

    # DISINI ITUNG SAWNYA
    # skalain jdi 1-5 dulu
    skor_npm = np.select([npm > 0.20, npm >= 0.12, npm >= 0.05, npm >= 0.01], [5, 4, 3, 2], default=1)
    skor_ato = np.select([atr >= 3.0, atr >= 2.0, atr >= 1.2, atr >= 0.7], [5, 4, 3, 2], default=1)
    skor_cfsr = np.select([cfsr <= 0.25, cfsr <= 0.40, cfsr <= 0.60, cfsr <= 0.85], [5, 4, 3, 2], default=1)
    skor_cr = np.select([cr >= 3.0, cr >= 2.0, cr >= 1.2, cr >= 1.0], [5, 4, 3, 2], default=1)
    cond_rg = [(rg >= 0.15) & (rg <= 0.40), (rg >= 0.05) & (rg < 0.15), (rg >= 0.0) & (rg < 0.05), ((rg >= -0.10) & (rg < 0.0)) | (rg > 0.40)]
    skor_rg = np.select(cond_rg, [5, 4, 3, 2], default=1)
    
    skor_oer = np.select([oer < 0.55, oer <= 0.70, oer <= 0.85, oer <= 0.95], [5, 4, 3, 2], default=1)

    # normalisasi (S - 1) / 4
    r_npm, r_ato, r_cfsr = (skor_npm - 1)/4, (skor_ato - 1)/4, (skor_cfsr - 1)/4
    r_cr, r_rg, r_oer    = (skor_cr - 1)/4, (skor_rg - 1)/4, (skor_oer - 1)/4

    # Kalkulasi Bobot ROC
    skor01 = (r_npm * 0.4083 + r_ato * 0.2417 + r_cfsr * 0.1583 + 
              r_cr * 0.1028 + r_rg * 0.0611 + r_oer * 0.0278)

    skor_kelayakan = round(float(skor01 * 100), 4)

    cursor.close()
    return features_dict, skor_kelayakan, id_pendapatan_terakhir