import pandas as pd

def create_results_table(results, file_name):
    """
    Salva os resultados em um arquivo de texto.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name, index=False, sep="\t")
    print(f"Resultados salvos em: {file_name}")
