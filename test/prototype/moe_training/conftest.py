import pytest, torch
@pytest.fixture(scope="session", autouse=True)
def _w():
    if torch.version.hip is None: return
    d = torch.zeros(2,2,dtype=torch.bfloat16,device="cuda")
    for _ in range(3):
        try: torch.mm(d,d); return
        except RuntimeError: pass
