import pulp
import random
import math
import time
import matplotlib.pyplot as plt


# Genera coordenadas aleatorias de ciudades y su matriz de distancias
def generate_tsp_instance(n_cities: int, seed: int = 42):
    random.seed(seed)
    coords = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(n_cities)}
    dist = {(i, j): math.dist(coords[i], coords[j]) for i in coords for j in coords if i != j}
    return coords, dist


# Crea una población inicial de rutas aleatorias para el algoritmo genético
def create_initial_population(pop_size: int, n_cities: int):
    base = list(range(n_cities))
    return [random.sample(base, len(base)) for _ in range(pop_size)]


# Calcula el costo total de una ruta según la matriz de distancias
def route_length(route, dist):
    return sum(dist.get((route[i], route[(i + 1) % len(route)]),
                        dist.get((route[(i + 1) % len(route)], route[i]), 0))
               for i in range(len(route)))


# Selección por torneo: escoge la mejor entre k rutas aleatorias
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(range(len(population))), k)
    selected.sort(key=lambda idx: fitnesses[idx])
    return population[selected[0]]


# Operador de cruce ordenado (preserva la estructura de las rutas)
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    p2_seq = [c for c in parent2 if c not in child]
    fill_positions = [i for i, v in enumerate(child) if v is None]
    for pos, val in zip(fill_positions, p2_seq):
        child[pos] = val
    return child


# Mutación simple: intercambia dos ciudades de la ruta
def swap_mutation(individual, p=0.1):
    ind = individual[:]
    if random.random() < p:
        a, b = random.sample(range(len(ind)), 2)
        ind[a], ind[b] = ind[b], ind[a]
    return ind


# Implementación completa del algoritmo genético para el TSP
def genetic_algorithm_tsp(dist, n_cities, pop_size=80, n_generations=300, mutation_prob=0.15, elite_size=5):
    start_time = time.time()
    population = create_initial_population(pop_size, n_cities)
    best_route, best_cost = None, float("inf")
    history = []

    for _ in range(n_generations):
        fitnesses = [route_length(ind, dist) for ind in population]

        # Guarda la mejor ruta encontrada hasta este punto
        for ind, fit in zip(population, fitnesses):
            if fit < best_cost:
                best_cost, best_route = fit, ind

        history.append(best_cost)

        # Ordena la población por calidad
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1])

        # Se conservan los mejores (elitismo)
        new_population = [ind for ind, _ in ranked[:elite_size]]

        # Se generan nuevos individuos por cruce y mutación
        while len(new_population) < pop_size:
            p1, p2 = tournament_selection(population, fitnesses), tournament_selection(population, fitnesses)
            child = ordered_crossover(p1, p2)
            new_population.append(swap_mutation(child, mutation_prob))

        population = new_population

    return {
        "best_route": best_route,
        "best_cost": best_cost,
        "time": time.time() - start_time,
        "history": history,
        "generations": n_generations,
    }


# Implementación del método exacto MTZ con programación lineal
def mtz_tsp(dist, n_cities, time_limit=None):
    start = time.time()
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    cities = list(range(n_cities))

    # Variables binarias x_ij y posiciones u_i
    x = pulp.LpVariable.dicts("x", (cities, cities), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", cities, 0, n_cities - 1)

    # Función objetivo: costo total del recorrido
    prob += pulp.lpSum(dist[i, j] * x[i][j] for i in cities for j in cities if i != j)

    # Cada ciudad tiene una entrada y una salida
    for i in cities:
        prob += pulp.lpSum(x[i][j] for j in cities if j != i) == 1
        prob += pulp.lpSum(x[j][i] for j in cities if j != i) == 1

    # Ciudad inicial con posición 0
    prob += u[0] == 0

    # Restricciones MTZ para evitar subtours
    for i in cities:
        for j in cities:
            if i != j and j != 0:
                prob += u[i] - u[j] + (n_cities - 1) * x[i][j] <= n_cities - 2

    # Resolver con límite de tiempo
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit or 40, msg=False))
    end = time.time()

    # Si no hay solución factible, regresamos None
    if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
        return {"status": pulp.LpStatus[prob.status], "best_cost": None, "route": None, "time": end - start}

    # Reconstrucción de la ruta final
    route, current, visited = [0], 0, {0}
    for _ in range(n_cities - 1):
        for j in cities:
            if j != current and pulp.value(x[current][j]) > 0.5 and j not in visited:
                route.append(j)
                visited.add(j)
                current = j
                break

    return {"status": pulp.LpStatus[prob.status], "best_cost": pulp.value(prob.objective), "route": route, "time": end - start}


# Encuentra en qué generación apareció la mejor solución
def generation_of_best(history):
    best = min(history)
    for i, v in enumerate(history):
        if v == best:
            return i + 1
    return len(history)


# Ejecución principal del experimento
if __name__ == "__main__":
    sizes = list(range(5, 65, 5))
    results = []

    for n in sizes:
        coords, dist = generate_tsp_instance(n, seed=2025 + n)
        ga_res = genetic_algorithm_tsp(dist, n, pop_size=90, n_generations=220)

        # Se aumenta el límite de tiempo para MTZ en instancias grandes
        tl = 40 if n <= 40 else (70 if n <= 50 else 90)
        mtz_res = mtz_tsp(dist, n, time_limit=tl)

        gap = (ga_res["best_cost"] - mtz_res["best_cost"]) / mtz_res["best_cost"] if mtz_res["best_cost"] else None

        # Se almacenan los resultados para graficarlos al final
        results.append({
            "n": n,
            "ga_cost": ga_res["best_cost"],
            "ga_time": ga_res["time"],
            "mtz_cost": mtz_res["best_cost"],
            "mtz_time": mtz_res["time"],
            "mtz_status": mtz_res["status"],
            "gap": gap,
            "ga_history": ga_res["history"],
            "best_gen": generation_of_best(ga_res["history"])
        })

        print(f"n={n} | GA: {ga_res['best_cost']:.2f} ({ga_res['time']:.2f}s) | "
              f"MTZ: {mtz_res['best_cost']} ({mtz_res['time']:.2f}s) [{mtz_res['status']}] | Gap={gap}")

        # Gráfica individual de convergencia por tamaño
        plt.figure()
        plt.plot(range(1, len(ga_res["history"]) + 1), ga_res["history"], label=f"GA ({n} nodos)")
        if mtz_res["best_cost"]:
            plt.plot(range(1, len(ga_res["history"]) + 1), [mtz_res["best_cost"]] * len(ga_res["history"]), label="MTZ (ref.)")
        plt.title(f"Convergencia para {n} ciudades (mejor en gen {generation_of_best(ga_res['history'])})")
        plt.xlabel("Generación")
        plt.ylabel("Costo")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Gráfica comparativa de tiempos
    plt.figure()
    plt.plot(sizes, [r["ga_time"] for r in results], marker="o", label="GA")
    plt.plot(sizes, [r["mtz_time"] for r in results], marker="o", label="MTZ")
    plt.title("Tiempo por método y número de ciudades")
    plt.xlabel("Número de ciudades")
    plt.ylabel("Tiempo (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfica de costos
    plt.figure()
    plt.plot(sizes, [r["ga_cost"] for r in results], marker="o", label="GA")
    plt.plot(sizes, [r["mtz_cost"] for r in results], marker="o", label="MTZ")
    plt.title("Costo total de ruta por método")
    plt.xlabel("Número de ciudades")
    plt.ylabel("Costo")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gap porcentual GA vs MTZ
    plt.figure()
    gaps = [r["gap"] * 100 if r["gap"] is not None else 0 for r in results]
    labels = [str(r["n"]) for r in results]
    colors = ["red" if (results[i]["gap"] is not None and gaps[i] > 0) else "green" for i in range(len(results))]
    plt.bar(labels, gaps, color=colors)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Gap GA vs MTZ (%)")
    plt.xlabel("Número de ciudades")
    plt.ylabel("Gap (%)")
    plt.grid(axis="y")
    plt.show()

