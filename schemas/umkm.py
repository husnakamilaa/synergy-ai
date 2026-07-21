from pydantic import BaseModel
from typing import Optional

class AnalisisRequest(BaseModel):
    id_umkm: str
    id_akad_variable: int

class AnalisisResponse(BaseModel):
    status: str
    id_analisis: int
    id_umkm: str
    skor_kelayakan: float
    akad: str
    message: str

# Schema tambahan untuk Response GET Detail
class DetailAnalisisData(BaseModel):
    id: int
    id_umkm: str
    id_akad_variable: int
    id_pendapatan: int
    current_ratio: float
    net_profit_margin: float
    operating_expense_ratio: float
    cashflow_stability_risk: float
    asset_turnover_ratio: float
    skor_kelayakan: float
    akad: str

class DetailAnalisisResponse(BaseModel):
    status: str
    data: DetailAnalisisData