import argparse
import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Calcula métricas y visualiza los resultados de la evaluación manual.")
    parser.add_argument("csv_path", type=Path, help="Ruta al archivo CSV con las notas manuales llenas.")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"No se encontró el archivo {args.csv_path}")
        return

    #dict[estrategia] = list[notas]
    notas_por_estrategia = defaultdict(list)

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            estrategia = row["estrategia"]
            nota_str = row["nota_manual_1_a_7"].strip()
            
            if nota_str:
                try:
                    nota = float(nota_str.replace(",", "."))
                    if 1.0 <= nota <= 7.0:
                        notas_por_estrategia[estrategia].append(nota)
                except ValueError:
                    pass

    if not notas_por_estrategia:
        print("No se encontraron notas válidas en el CSV. Asegúrate de llenar la columna 'nota_manual_1_a_7'.")
        return

    #calculamos promedios
    resultados = {}
    print("\n=== RESULTADOS DE EVALUACIÓN MANUAL (Promedio de Pertinencia 1-7) ===")
    for estrategia, notas in sorted(notas_por_estrategia.items()):
        promedio = sum(notas) / len(notas)
        resultados[estrategia] = promedio
        print(f"{estrategia}: {promedio:.2f} (basado en {len(notas)} chunks evaluados)")

    #VISUALIZACION
    estrategias = list(resultados.keys())
    promedios = list(resultados.values())

    plt.figure(figsize=(10, 6))
    colores = ['#4A90E2', '#50E3C2', '#F5A623', '#F8E71C']
    barras = plt.bar(estrategias, promedios, color=colores[:len(estrategias)])
    
    plt.ylim(1, 7)
    plt.title("Relevancia Promedio por Estrategia de Recuperación (Evaluación Manual)", fontsize=14, pad=15)
    plt.ylabel("Nota Promedio (1 a 7)", fontsize=12)
    plt.xlabel("Configuración de Recuperación", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    ruta_grafico = args.csv_path.parent / "comparacion_metricas.png"
    plt.savefig(ruta_grafico, bbox_inches='tight', dpi=300)
    print(f"\n[Éxito] Gráfico guardado en: {ruta_grafico}")
    plt.show()

if __name__ == "__main__":
    main()