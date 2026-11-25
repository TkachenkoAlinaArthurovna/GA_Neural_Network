import torch
import random
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from statistics import mean, pstdev
from pathlib import Path
import csv
import json


# Завантажуємо датасет
dataset = load_breast_cancer()

# Вхідні ознаки
X = dataset.data

# Правильні відповіді
y = dataset.target

print("Кількість прикладів:", X.shape[0])
print("Кількість ознак:", X.shape[1])

print("\nНазви перших п’яти ознак:")
for name in dataset.feature_names[:5]:
    print(name)

print("\nПерші п’ять ознак першого прикладу:")
print(X[0, :5])

print("\nПравильна відповідь для першого прикладу:")
print(y[0])

print("\nНазви класів:")
for number, name in enumerate(dataset.target_names):
    print(number, "—", name)


# Відділяємо 20% даних для остаточного тестування
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# З решти 80% виділяємо частину для перевірки кандидатів
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print("\nРозподіл даних:")
print("Навчальна вибірка:", X_train.shape)
print("Валідаційна вибірка:", X_val.shape)
print("Тестова вибірка:", X_test.shape)

print("\nКількість відповідей:")
print("Навчальні:", len(y_train))
print("Валідаційні:", len(y_val))
print("Тестові:", len(y_test))


# Подивимося на дані до масштабування
print("\nПерші 5 ознак до масштабування:")
print(X_train[0, :5])

# Створюємо об’єкт масштабування
scaler = StandardScaler()

# Обчислюємо параметри лише на навчальній вибірці
# та перетворюємо її
X_train = scaler.fit_transform(X_train)

# Застосовуємо ті самі параметри до інших вибірок
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nПерші 5 ознак після масштабування:")
print(X_train[0, :5])

print("\nСередні значення перших 5 навчальних ознак:")
print(X_train.mean(axis=0)[:5])

print("\nСтандартні відхилення перших 5 навчальних ознак:")
print(X_train.std(axis=0)[:5])


# Перетворюємо ознаки на тензори з дробовими числами
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

# Перетворюємо правильні відповіді на цілочисельні тензори
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

print("\nНавчальні ознаки:")
print("Розмір:", X_train_tensor.shape)
print("Тип:", X_train_tensor.dtype)

print("\nНавчальні відповіді:")
print("Розмір:", y_train_tensor.shape)
print("Тип:", y_train_tensor.dtype)

print("\nПерші 5 навчальних відповідей:")
print(y_train_tensor[:5])


def create_model(
    input_size,
    hidden_sizes=(16,),
    activation="relu",
    output_size=2
):
    layers = []
    previous_size = input_size

    for hidden_size in hidden_sizes:
        # Шар із вагами та зміщеннями
        layers.append(
            nn.Linear(previous_size, hidden_size)
        )

        # Функція активації після прихованого шару
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError("Невідома функція активації")

        previous_size = hidden_size

    # Вихідний шар
    layers.append(
        nn.Linear(previous_size, output_size)
    )

    return nn.Sequential(*layers)


# Фіксуємо генератор початкових ваг
torch.manual_seed(42)

baseline_model = create_model(
    input_size=30,
    hidden_sizes=(16,),
    activation="relu",
    output_size=2
)

print("\nАрхітектура базової моделі:")
print(baseline_model)

parameter_count = sum(
    parameter.numel()
    for parameter in baseline_model.parameters()
)

print("\nКількість параметрів:", parameter_count)

# Перевіряємо модель на перших трьох прикладах
with torch.no_grad():
    outputs = baseline_model(X_train_tensor[:3])

print("\nРозмір виходу:", outputs.shape)
print("Виходи ненавченої мережі:")
print(outputs)


def train_model(
    model,
    X,
    y,
    epochs=60,
    learning_rate=0.001,
    verbose=True
):
    # Функція, яка вимірює помилку класифікації
    loss_function = nn.CrossEntropyLoss()

    # Оптимізатор, який оновлює ваги та зміщення
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_history = []

    # Переводимо модель у режим навчання
    model.train()

    for epoch in range(epochs):
        # 1. Очищаємо попередні градієнти
        optimizer.zero_grad()

        # 2. Обчислюємо виходи мережі
        outputs = model(X)

        # 3. Порівнюємо виходи з правильними відповідями
        loss = loss_function(outputs, y)

        # 4. Обчислюємо градієнти помилки
        loss.backward()

        # 5. Оновлюємо параметри моделі
        optimizer.step()

        # Зберігаємо числове значення помилки
        loss_history.append(loss.item())

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Епоха {epoch + 1}/{epochs}, "
                f"Loss: {loss.item():.4f}"
            )

    return loss_history


print("\nНавчання базової моделі:")

baseline_history = train_model(
    model=baseline_model,
    X=X_train_tensor,
    y=y_train_tensor,
    epochs=60,
    learning_rate=0.001,
    verbose=True
)

print("\nНавчання завершено!")
print(f"Loss першої епохи: {baseline_history[0]:.4f}")
print(f"Loss останньої епохи: {baseline_history[-1]:.4f}")


def evaluate_model(model, X, y):
    # Переводимо мережу в режим оцінювання
    model.eval()

    # Під час оцінювання градієнти не потрібні
    with torch.no_grad():
        outputs = model(X)

        # Вибираємо клас із найбільшою оцінкою
        predictions = outputs.argmax(dim=1)

        # Обчислюємо частку правильних відповідей
        accuracy = (
            (predictions == y)
            .float()
            .mean()
            .item()
        )

    return accuracy


baseline_train_accuracy = evaluate_model(
    baseline_model,
    X_train_tensor,
    y_train_tensor
)

baseline_val_accuracy = evaluate_model(
    baseline_model,
    X_val_tensor,
    y_val_tensor
)

print(
    f"\nНавчальна точність: "
    f"{baseline_train_accuracy:.2%}"
)

print(
    f"Валідаційна точність: "
    f"{baseline_val_accuracy:.2%}"
)

# Подивимося на перші 10 валідаційних прогнозів
with torch.no_grad():
    val_outputs = baseline_model(X_val_tensor)
    val_predictions = val_outputs.argmax(dim=1)

print("\nПерші 10 прогнозів:")
print(val_predictions[:10])

print("Перші 10 правильних відповідей:")
print(y_val_tensor[:10])

correct_count = (
    val_predictions == y_val_tensor
).sum().item()

print(
    f"\nПравильних відповідей на валідації: "
    f"{correct_count} із {len(y_val_tensor)}"
)


# Варіанти гіперпараметрів для генетичного алгоритму
NEURON_OPTIONS = [8, 16, 32, 64]

ACTIVATION_OPTIONS = ["relu", "tanh"]

LEARNING_RATE_OPTIONS = [0.001, 0.003, 0.01, 0.03]


def decode_individual(individual):
    # Перший ген — кількість прихованих шарів
    layer_count = individual[0]

    # Наступні три гени — індекси кількості нейронів
    hidden_sizes = tuple(
        NEURON_OPTIONS[individual[i + 1]]
        for i in range(layer_count)
    )

    # П’ятий ген — індекс активації
    activation = ACTIVATION_OPTIONS[individual[4]]

    # Шостий ген — індекс швидкості навчання
    learning_rate = LEARNING_RATE_OPTIONS[individual[5]]

    return hidden_sizes, activation, learning_rate
example_individual = [2, 2, 1, 0, 0, 3]

hidden_sizes, activation, learning_rate = decode_individual(
    example_individual
)

print("\nПриклад хромосоми:", example_individual)
print("Приховані шари:", hidden_sizes)
print("Активація:", activation)
print("Швидкість навчання:", learning_rate)


# Однакова кількість епох для всіх кандидатів
SEARCH_EPOCHS = 60

# Зберігаємо оцінки вже перевірених конфігурацій
fitness_cache = {}


def evaluate_individual(individual):
    hidden_sizes, activation, learning_rate = decode_individual(
        individual
    )

    # Ключ описує фактичну конфігурацію мережі
    configuration = (
        hidden_sizes,
        activation,
        learning_rate
    )

    # Якщо таку мережу вже оцінювали, повертаємо результат
    if configuration in fitness_cache:
        return fitness_cache[configuration]

    # Фіксуємо випадкову ініціалізацію
    torch.manual_seed(42)

    # Кожну нову конфігурацію навчаємо з нуля
    model = create_model(
        input_size=X_train_tensor.shape[1],
        hidden_sizes=hidden_sizes,
        activation=activation,
        output_size=2
    )

    train_model(
        model=model,
        X=X_train_tensor,
        y=y_train_tensor,
        epochs=SEARCH_EPOCHS,
        learning_rate=learning_rate,
        verbose=False
    )

    accuracy = evaluate_model(
        model,
        X_val_tensor,
        y_val_tensor
    )

    # DEAP очікує кортеж оцінок
    fitness = (accuracy,)

    fitness_cache[configuration] = fitness

    return fitness


# 1 прихований шар, 16 нейронів, ReLU, learning_rate=0.001
baseline_individual = [1, 1, 1, 1, 0, 0]

baseline_fitness = evaluate_individual(
    baseline_individual
)

print("\nFitness базової конфігурації:", baseline_fitness)

print(
    f"Валідаційна точність кандидата: "
    f"{baseline_fitness[0]:.2%}"
)

print(
    "Кількість оцінених конфігурацій:",
    len(fitness_cache)
)


# Фіксуємо випадкову генерацію хромосом
random.seed(42)

POPULATION_SIZE = 20

# Допустимі значення кожного гена
GENE_OPTIONS = [
    [1, 2, 3],       # кількість прихованих шарів
    [0, 1, 2, 3],    # індекс ширини першого шару
    [0, 1, 2, 3],    # індекс ширини другого шару
    [0, 1, 2, 3],    # індекс ширини третього шару
    [0, 1],          # індекс активації
    [0, 1, 2, 3]     # індекс швидкості навчання
]


# Додатна вага означає: більша точність — краще
creator.create(
    "FitnessMax",
    base.Fitness,
    weights=(1.0,)
)

# Особина — список генів, який також має fitness
creator.create(
    "Individual",
    list,
    fitness=creator.FitnessMax
)


def generate_individual():
    return [
        random.choice(options)
        for options in GENE_OPTIONS
    ]


# Набір операцій генетичного алгоритму
toolbox = base.Toolbox()

# Створення однієї особини
toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    generate_individual
)

# Створення списку особин — популяції
toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

# Оцінювання особини нашою fitness-функцією
toolbox.register(
    "evaluate",
    evaluate_individual
)


population = toolbox.population(n=POPULATION_SIZE)

# Одну особину замінюємо базовою конфігурацією
population[0][:] = baseline_individual

print("\nРозмір популяції:", len(population))

print("\nПерші 5 особин:")

for number, individual in enumerate(population[:5], start=1):
    hidden_sizes, activation, learning_rate = decode_individual(
        individual
    )

    print(f"\nОсобина {number}: {list(individual)}")
    print("Приховані шари:", hidden_sizes)
    print("Активація:", activation)
    print("Швидкість навчання:", learning_rate)
    print("Fitness обчислено:", individual.fitness.valid)


def mutate_individual(individual, gene_probability=0.2):
    for index, options in enumerate(GENE_OPTIONS):
        # Для кожного гена вирішуємо, чи змінювати його
        if random.random() < gene_probability:
            # Вибираємо лише значення, відмінні від поточного
            alternatives = [
                value
                for value in options
                if value != individual[index]
            ]

            individual[index] = random.choice(alternatives)

    # DEAP очікує кортеж змінених особин
    return (individual,)


# Турнірний відбір
toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

# Двоточковий кросовер
toolbox.register(
    "mate",
    tools.cxTwoPoint
)

# Мутація
toolbox.register(
    "mutate",
    mutate_individual,
    gene_probability=0.2
)

print("\nОператори генетичного алгоритму зареєстровані.")


# Зберігаємо стан генератора, щоб демонстрація
# не змінювала випадкову послідовність майбутньої еволюції
saved_random_state = random.getstate()

parent_a = toolbox.clone(population[0])
parent_b = toolbox.clone(population[1])

print("\nДо кросоверу:")
print("A:", list(parent_a))
print("B:", list(parent_b))

toolbox.mate(parent_a, parent_b)

print("\nПісля кросоверу:")
print("A:", list(parent_a))
print("B:", list(parent_b))

before_mutation = list(parent_a)

toolbox.mutate(parent_a)

print("\nМутація першого нащадка:")
print("До:", before_mutation)
print("Після:", list(parent_a))

random.setstate(saved_random_state)


GENERATIONS = 50

# Для маленьких мереж використовуємо один CPU-потік
torch.set_num_threads(1)

# Зберігаємо найкращу особину за весь пошук
hall_of_fame = tools.HallOfFame(1)

# Вказуємо, який показник збираємо з кожної особини
statistics = tools.Statistics(
    lambda individual: individual.fitness.values[0]
)

statistics.register(
    "avg",
    lambda values: sum(values) / len(values)
)

statistics.register("max", max)
statistics.register("min", min)


print("\nЗапуск генетичного алгоритму:")

population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,
    mutpb=0.3,
    ngen=GENERATIONS,
    stats=statistics,
    halloffame=hall_of_fame,
    verbose=True
)


best_individual = hall_of_fame[0]

best_hidden_sizes, best_activation, best_learning_rate = (
    decode_individual(best_individual)
)

best_val_accuracy = best_individual.fitness.values[0]

print("\nРезультат генетичного алгоритму:")

print("Найкраща хромосома:", list(best_individual))
print("Приховані шари:", best_hidden_sizes)
print("Активація:", best_activation)
print("Швидкість навчання:", best_learning_rate)

print(
    f"\nБазова валідаційна точність: "
    f"{baseline_val_accuracy:.2%}"
)

print(
    f"Найкраща валідаційна точність: "
    f"{best_val_accuracy:.2%}"
)

difference_pp = (
    best_val_accuracy - baseline_val_accuracy
) * 100

print(
    f"Різниця: {difference_pp:+.2f} "
    "відсоткового пункту"
)

print(
    "Унікальних оцінених конфігурацій:",
    len(fitness_cache)
)


# Відновлюємо той самий зовнішній поділ 80/20.
# Однакові random_state і stratify дають ті самі приклади.
X_final_train_raw, X_final_test_raw, y_final_train, y_final_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# Для фінального навчання обчислюємо масштабування
# на всіх 455 навчальних прикладах
final_scaler = StandardScaler()

X_final_train = final_scaler.fit_transform(
    X_final_train_raw
)

X_final_test = final_scaler.transform(
    X_final_test_raw
)

X_final_train_tensor = torch.tensor(
    X_final_train,
    dtype=torch.float32
)

X_final_test_tensor = torch.tensor(
    X_final_test,
    dtype=torch.float32
)

y_final_train_tensor = torch.tensor(
    y_final_train,
    dtype=torch.long
)

y_final_test_tensor = torch.tensor(
    y_final_test,
    dtype=torch.long
)

print("\nФінальне навчання:")
print("Навчальних прикладів:", len(y_final_train))
print("Тестових прикладів:", len(y_final_test))


FINAL_EPOCHS = 60
FINAL_SEEDS = [303, 404, 505]

configurations = {
    "Базова": {
        "hidden_sizes": (16,),
        "activation": "relu",
        "learning_rate": 0.001
    },
    "Оптимізована": {
        "hidden_sizes": best_hidden_sizes,
        "activation": best_activation,
        "learning_rate": best_learning_rate
    }
}

test_results = {}

for model_name, parameters in configurations.items():
    accuracies = []

    print(f"\nМодель: {model_name}")

    for seed in FINAL_SEEDS:
        torch.manual_seed(seed)

        # Створюємо нову модель з початковими вагами
        final_model = create_model(
            input_size=30,
            hidden_sizes=parameters["hidden_sizes"],
            activation=parameters["activation"],
            output_size=2
        )

        train_model(
            model=final_model,
            X=X_final_train_tensor,
            y=y_final_train_tensor,
            epochs=FINAL_EPOCHS,
            learning_rate=parameters["learning_rate"],
            verbose=False
        )

        test_accuracy = evaluate_model(
            final_model,
            X_final_test_tensor,
            y_final_test_tensor
        )

        accuracies.append(test_accuracy)

        print(
            f"Seed {seed}: "
            f"тестова точність = {test_accuracy:.2%}"
        )

    test_results[model_name] = accuracies

    print(
        f"Середня точність: "
        f"{mean(accuracies):.2%}"
    )

    print(
        f"Стандартне відхилення: "
        f"{pstdev(accuracies) * 100:.2f} "
        "відсоткового пункту"
    )

baseline_test_mean = mean(test_results["Базова"])
optimized_test_mean = mean(test_results["Оптимізована"])

test_difference_pp = (
    optimized_test_mean - baseline_test_mean
) * 100

print("\nПідсумкове порівняння:")

print(
    f"Базова модель: "
    f"{baseline_test_mean:.2%}"
)

print(
    f"Оптимізована модель: "
    f"{optimized_test_mean:.2%}"
)

print(
    f"Різниця: {test_difference_pp:+.2f} "
    "відсоткового пункту"
)


# Папка results поруч із main.py
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(baseline_history) + 1),
    baseline_history
)

plt.xlabel("Епоха")
plt.ylabel("Функція втрат")
plt.title("Навчання базової нейронної мережі")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "baseline_training.png",
    dpi=200
)

plt.close()


generations = logbook.select("gen")
average_fitness = logbook.select("avg")
maximum_fitness = logbook.select("max")

# Найкращий результат, отриманий до поточного покоління включно
best_so_far = []
current_best = 0.0

for value in maximum_fitness:
    current_best = max(current_best, value)
    best_so_far.append(current_best)

plt.figure(figsize=(8, 5))

plt.plot(
    generations,
    [value * 100 for value in average_fitness],
    label="Середня точність популяції"
)

plt.plot(
    generations,
    [value * 100 for value in best_so_far],
    label="Найкраща точність за весь пошук"
)

plt.xlabel("Покоління")
plt.ylabel("Валідаційна точність, %")
plt.title("Процес генетичної оптимізації")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "genetic_evolution.png",
    dpi=200
)

plt.close()


model_names = ["Базова", "Оптимізована"]

mean_percentages = [
    mean(test_results[name]) * 100
    for name in model_names
]

std_percentages = [
    pstdev(test_results[name]) * 100
    for name in model_names
]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    model_names,
    mean_percentages,
    yerr=std_percentages,
    capsize=8,
    color=["steelblue", "seagreen"],
    alpha=0.85
)

for bar, value in zip(bars, mean_percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        2,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        color="white",
        fontsize=12
    )

plt.ylabel("Тестова точність, %")
plt.ylim(0, 100)
plt.title("Середня точність за трьома запусками")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "model_comparison.png",
    dpi=200
)

plt.close()


# Точність кожного фінального запуску
with open(
    results_dir / "test_comparison.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "Модель",
        "Seed",
        "Тестова точність, %"
    ])

    for model_name in model_names:
        for seed, accuracy in zip(
            FINAL_SEEDS,
            test_results[model_name]
        ):
            writer.writerow([
                model_name,
                seed,
                accuracy * 100
            ])

# Статистика всіх поколінь
with open(
    results_dir / "evolution.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "Покоління",
        "Запити оцінювання",
        "Середній fitness",
        "Максимальний fitness",
        "Мінімальний fitness"
    ])

    for record in logbook:
        writer.writerow([
            record["gen"],
            record["nevals"],
            record["avg"],
            record["max"],
            record["min"]
        ])

# Підсумок експерименту
summary = {
    "best_hyperparameters": {
        "hidden_sizes": list(best_hidden_sizes),
        "activation": best_activation,
        "learning_rate": best_learning_rate
    },
    "baseline_validation_accuracy": baseline_val_accuracy,
    "best_validation_accuracy": best_val_accuracy,
    "unique_configurations": len(fitness_cache),
    "final_epochs": FINAL_EPOCHS,
    "final_seeds": FINAL_SEEDS,
    "test_results": test_results,
    "baseline_test_mean": baseline_test_mean,
    "optimized_test_mean": optimized_test_mean,
    "difference_percentage_points": test_difference_pp
}

with open(
    results_dir / "summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=2
    )

print("\nГрафіки й таблиці збережено:")
print(results_dir)