from tabulate import tabulate
import numpy as np


def create_results_table(results, filename):
    """Cria uma tabela com os resultados médios e salva em um arquivo."""

    # Cabeçalho da tabela
    table = [["K", "Distance", "Accuracy", "AUC"]]

    # Adicionando os dados de cada item de results
    for result in results:
        # Garantir que AUC é um valor numérico, caso seja np.float64
        auc_value = (
            float(result["AUC"])
            if isinstance(result["AUC"], np.float64)
            else result["AUC"]
        )

        # Adicionando a linha na tabela
        table.append([result["K"], result["Distance"], result["Accuracy"], auc_value])

    # Salvando a tabela em um arquivo
    with open(filename, "w") as file:
        file.write(tabulate(table, headers="firstrow", tablefmt="grid"))
