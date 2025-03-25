import itertools

def create_graph():
    """Creates the adjacency list representation of the TSP graph."""
    return {
        0: {1: 12, 2: 10, 3: 8, 4: 12},
        1: {0: 12, 3: 3, 5: 9, 6: 11},
        2: {0: 10, 4: 6, 5: 7},
        3: {0: 8, 1: 3, 5: 9},
        4: {0: 12, 2: 6, 6: 9},
        5: {1: 9, 2: 7, 3: 9, 6: 10},
        6: {1: 11, 4: 9, 5: 10}
    }

def tsp_dynamic_programming(graph):
    """Solves TSP using Dynamic Programming (Held-Karp algorithm) with adjacency list."""
    n = len(graph)
    all_sets = (1 << n) - 1
    
    memo = {}
    parent = {}
    
    def visit(city, visited):
        if visited == all_sets:
            return graph[city].get(0, float('inf'))  # Return to start if possible
        
        if (city, visited) in memo:
            return memo[(city, visited)]
        
        min_cost = float('inf')
        best_next = None
        for next_city in graph[city]:
            if visited & (1 << next_city) == 0:
                cost = graph[city][next_city] + visit(next_city, visited | (1 << next_city))
                if cost < min_cost:
                    min_cost = cost
                    best_next = next_city
        
        memo[(city, visited)] = min_cost
        parent[(city, visited)] = best_next
        return min_cost
    
    best_distance = visit(0, 1)
    
    route = [0]
    visited = 1
    current = 0
    while len(route) < n:
        next_city = parent.get((current, visited), None)
        if next_city is None:
            break
        route.append(next_city)
        visited |= (1 << next_city)
        current = next_city
    route.append(0)
    
    return best_distance, route

graph = create_graph()
best_distance_dp, best_route_dp = tsp_dynamic_programming(graph)

print("Optimal TSP route distance (DP):", best_distance_dp)
print("Optimal TSP route (DP):", best_route_dp)
