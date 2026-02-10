import pandas as pd

def load_biomedgraphica(csv_path):
    df = pd.read_csv(
        csv_path,
        usecols=[
            "BioMedGraphica_Conn_ID",
            "HGNC_Symbol",
            "Ensembl_Gene_ID",
        ]
    )

    hgnc_to_conn = {}
    alias_to_hgnc = {}

    for _, row in df.iterrows():
        hgnc = str(row["HGNC_Symbol"]).strip()
        conn = str(row["BioMedGraphica_Conn_ID"]).strip()
        if not hgnc or hgnc == "nan":
            continue

        hgnc_to_conn[hgnc] = conn

        # aliases
        alias_to_hgnc[hgnc] = hgnc
        alias_to_hgnc[hgnc.upper()] = hgnc
        alias_to_hgnc[hgnc.lower()] = hgnc

        if pd.notna(row["Ensembl_Gene_ID"]):
            for eid in str(row["Ensembl_Gene_ID"]).split(";"):
                alias_to_hgnc[eid.strip()] = hgnc

    return hgnc_to_conn, alias_to_hgnc



