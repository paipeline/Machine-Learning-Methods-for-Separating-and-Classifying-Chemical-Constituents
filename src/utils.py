import matplotlib.pyplot as plt

def plot_metrics(metric_name, data):
    plt.figure(figsize=(10, 6))
    for model_name, metric in data.items():
        plt.plot(metric[metric_name], label=f"{model_name} - {metric_name}")
    plt.title(f"{metric_name} for models")
    plt.xlabel("folds")
    plt.ylabel(metric_name)
    plt.grid()
    plt.legend()
    plt.savefig(f"{metric_name}.png")
    plt.show()

def print_metrics(mae, mse, model_name, r2, err):
    print("==================================")
    print(f"model: {model_name}")
    print(f"mae avg: {mae}")
    print(f"r2 avg: {r2}")
    print(f"mse avg: {mse}")
    print(f"Error Rate: {err}")
