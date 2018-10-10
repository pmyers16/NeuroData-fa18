# from __future__ import print_function
map = {}

with open('barbell.edgelist', 'r') as graph:
    for edge in graph:

        edge = [int(x) for x in edge.strip('\n').split(' ')]

        if edge[0] in map.keys():
            map[edge[0]].append(edge[1])
        else:
            map[edge[0]] = [edge[1]]

        if edge[1] in map.keys():
            if edge[0] not in map[edge[1]]:
                map[edge[1]].append(edge[0])
        else:
            map[edge[1]] = [edge[0]]


def neighbors(u):
    return map[u]


neighborhood = {}
# 1-hop neighbors
for u in map.keys():
    neighborhood[u] = [len(neighbors(u))]


# 2-hop neighbors
for u in map.keys():
    num_neighbors = len(neighbors(u))
    for v in neighbors(u):
        num_neighbors += len(neighbors(v))

    neighborhood[u].append(num_neighbors)

# def k_locality(v, k, accumulated_neighbors=0):
#
#     if k == 0:
#         return 1
#     elif k == 1:
#         accumulated_neighbors += len(neighbors(v))
#         return accumulated_neighbors
#
#     else:
#         for u in neighbors(v):
#             print(k_locality(u, k-1, accumulated_neighbors))

# print(map)
# print(k_locality(0, 2))

print(neighborhood)

import matplotlib.pyplot as plt
xs = [neighborhood[u][0] for u in neighborhood.keys()]
ys = [neighborhood[u][1] for u in neighborhood.keys()]
plt.scatter(xs, ys, alpha=0.5)
plt.show()
