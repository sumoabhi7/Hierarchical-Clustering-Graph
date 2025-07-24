import networkx as nx

G = nx.Graph()
edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('D', 'E')]
G.add_edges_from(edges)

# Get neighbors
neighbors_A = set(G.neighbors('A'))  # {'B', 'D'}
neighbors_E = set(G.neighbors('D'))  # {'B', 'D'}

# Jaccard Similarity
intersection = len(neighbors_A & neighbors_E)  # 2
union = len(neighbors_A | neighbors_E)        # 2
similarity = intersection / union             # 1.0
distance = 1 - similarity                    # 0.0

print(f"A-to-E Distance: {distance}")  # Output: 0.0