import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


def load_file(file_path):
    if not os.path.exists(file_path):
        print("Ошибка: Файл не найден!")
        return None

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return None


def cluster_data(df):
    features = df.select_dtypes(include=['number'])

    if features.empty:
        print("Ошибка: В файле нет числовых данных для кластеризации")
        return None

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(features)
    return df


def show_plot(df):
    plt.figure(figsize=(6, 4))
    plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c=df['Segment'], cmap='viridis')
    plt.xlabel(df.columns[1])
    plt.ylabel(df.columns[2])
    plt.title("Кластеризация клиентов")
    plt.show()


def main():
    file_path = "clients.csv"  # Файл должен быть в папке "data"
    df = load_file(file_path)

    if df is not None:
        df = cluster_data(df)
        if df is not None:
            print("Результаты кластеризации:")
            print(df[['ID', 'Segment']])
            show_plot(df)


if __name__ == "__main__":
    main()