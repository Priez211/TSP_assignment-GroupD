import itertools

def create_graph():
    """Creates the adjacency list representation of the TSP graph."""
    return {
        1: {2:12, 3:10, 7:12}, 
        2: {1:12, 3:8, 4:12}, 
        3: {1:10, 2:8, 4:11, 5:3, 7:9}, 
        4: {2:12, 3:11, 5:11, 6:10},
        5: {3:3, 4:11, 6:6, 7:7},
        6: {4:10, 5:6, 7:9},
        7: {1:12, 3:9, 5:7, 6:9},
    }

def tsp_dynamic_programming(graph):
    """Solves TSP using Dynamic Programming (Held-Karp algorithm) with adjacency list."""
    cities = sorted(graph.keys())
    n = len(cities)
    city_to_index = {city: idx for idx, city in enumerate(cities)}
    all_sets = (1 << n) - 1
    
    memo = {}
    parent = {}
    
    def visit(city_idx, visited):
        if visited == all_sets:
            # Return to start city (city 1)
            original_city = cities[city_idx]
            start_city = cities[0]
            return graph[original_city].get(start_city, float('inf'))
        
        if (city_idx, visited) in memo:
            return memo[(city_idx, visited)]
        
        min_cost = float('inf')
        best_next = None
        original_city = cities[city_idx]
        
        for neighbor in graph[original_city]:
            neighbor_idx = city_to_index[neighbor]
            if visited & (1 << neighbor_idx) == 0:
                cost = graph[original_city][neighbor] + visit(neighbor_idx, visited | (1 << neighbor_idx))
                if cost < min_cost:
                    min_cost = cost
                    best_next = neighbor_idx
        
        memo[(city_idx, visited)] = min_cost
        parent[(city_idx, visited)] = best_next
        return min_cost
    
    # Start from city 1 (index 0)
    best_distance = visit(0, 1)
    
    # Reconstruct the path
    route_indices = [0]
    visited = 1
    current_idx = 0
    
    while len(route_indices) < n:
        next_idx = parent.get((current_idx, visited), None)
        if next_idx is None:
            break
        route_indices.append(next_idx)
        visited |= (1 << next_idx)
        current_idx = next_idx
    
    # Convert indices back to city numbers
    best_route = [cities[idx] for idx in route_indices]
    best_route.append(cities[0])  # Return to start
    
    return best_distance, best_route

graph = create_graph()
best_distance_dp, best_route_dp = tsp_dynamic_programming(graph)

print("Optimal TSP route distance (DP):", best_distance_dp)
print("Optimal TSP route (DP):", best_route_dp)
