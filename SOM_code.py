import numpy as np

def som_tsp_adjacency(graph, num_neurons=30, iterations=10000, lr=0.9):
    
    graph = {int(k): {int(k2): float(v) for k2, v in v.items()} for k, v in graph.items()}
    
    cities = list(graph.keys())
    start_city = 1  
    
    neurons = []
    for _ in range(num_neurons):
        path = nearest_neighbor_path(graph, start_city)
        neurons.append(path)
    
    for _ in range(num_neurons // 2):
        while True:
            middle_cities = [c for c in cities if c != start_city]
            candidate = [start_city] + list(np.random.permutation(middle_cities)) + [start_city]
            if is_valid_path(candidate, graph):
                neurons.append(candidate)
                break
    
    best_path = None
    best_dist = float('inf')
    
    for i in range(iterations):
        lr_decay = lr * (1 - i / iterations)
        sigma_decay = num_neurons * (1 - i / iterations)
        
        for city in np.random.permutation(cities):
            winner, _ = find_winner(neurons, city, graph)
            
            for j in range(num_neurons):
                influence = calculate_influence(j, winner, sigma_decay, num_neurons)
                if influence > 0.2:
                    neurons[j] = mutate_preserving_start(neurons[j], graph, lr_decay, start_city)
        
        # Updating best solution
        current_path, current_dist = find_best_path(neurons, graph, start_city)
        if current_dist < best_dist:
            best_dist = current_dist
            best_path = current_path
    
    if best_path[0] != start_city or best_path[-1] != start_city:
        best_path = [start_city] + [c for c in best_path if c != start_city] + [start_city]
        best_dist = calculate_path_distance(best_path, graph)
    
    return best_path, best_dist

def is_valid_path(path, graph):
    #Checking if all consecutive cities in path are connected
    for i in range(len(path)-1):
        if path[i+1] not in graph[path[i]]:
            return False
    return True

def nearest_neighbor_path(graph, start_city):
    cities = list(graph.keys())
    unvisited = set(cities)
    unvisited.remove(start_city)
    path = [start_city]
    
    while unvisited:
        last = path[-1]
        nearest = min(unvisited, key=lambda x: graph[last].get(x, float('inf')))
        path.append(nearest)
        unvisited.remove(nearest)
    
    path.append(start_city)  # Return to start
    return path

def find_winner(neurons, city, graph):
    min_dist = float('inf')
    winner = 0
    for n_idx, neuron in enumerate(neurons):
        try:
            pos = neuron.index(city)
            prev_c = neuron[pos-1] if pos > 0 else neuron[-1]
            next_c = neuron[pos+1] if pos < len(neuron)-1 else neuron[0]
            dist = graph[prev_c].get(city, float('inf')) + graph[city].get(next_c, float('inf'))
            if dist < min_dist:
                min_dist = dist
                winner = n_idx
        except ValueError:
            continue
    return winner, min_dist

def calculate_influence(j, winner, sigma_decay, num_neurons):
    dist = min(abs(j - winner), num_neurons - abs(j - winner))
    return np.exp(-(dist ** 2) / (2 * sigma_decay ** 2))

def mutate_preserving_start(path, graph, mutation_rate, start_city):
    #Mutation that keeps first/last as start_city
    if len(path) <= 3:  
        return path
    
    new_path = path.copy()
    
    if np.random.random() < mutation_rate * 0.3:
        i = np.random.randint(1, len(path)-2)
        j = np.random.randint(i+1, len(path)-1)
        new_path[i:j+1] = new_path[i:j+1][::-1]
        if is_valid_path(new_path, graph):
            return new_path
    
    if np.random.random() < mutation_rate:
        i = np.random.randint(1, len(path)-1)
        j = np.random.randint(1, len(path)-1)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        if is_valid_path(new_path, graph):
            return new_path
    
    return path

def find_best_path(neurons, graph, start_city):
    best_path = None
    best_dist = float('inf')
    
    for neuron in neurons:
        if not is_valid_path(neuron, graph) or neuron[0] != start_city or neuron[-1] != start_city:
            continue
        
        current_dist = calculate_path_distance(neuron, graph)
        if current_dist < best_dist:
            best_dist = current_dist
            best_path = neuron
    
    return best_path, best_dist

def calculate_path_distance(path, graph):
    """Calculate total distance of a path"""
    distance = 0
    for i in range(len(path)-1):
        distance += graph[path[i]][path[i+1]]
    return distance

graph = {
    1: {2:12, 3:10, 7:12}, 
    2: {1:12, 3:8, 4:12}, 
    3: {1:10, 2:8, 4:11, 5:3, 7:9}, 
    4: {2:12, 3:11, 5:11, 6:10},
    5: {3:3, 4:11, 6:6, 7:7},
    6: {4:10, 5:6, 7:9},
    7: {1:12, 3:9, 5:7, 6:9},
}

best_path, best_dist = som_tsp_adjacency(graph)
print("Optimal Path:", best_path)
print("Total Distance:", best_dist)