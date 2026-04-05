Hierarchical Clustering
    Clustering is a way of grouping similar things together.
    Example: If you had a basket with apples, oranges, and bananas, clustering would mean sorting them into groups so that each group has the same type of fruit.
    In data science:-
        A data point = a single object/item with certain features.
        Cluster = a group of similar data points.
    The word hierarchical means “arranged in levels” — like a family tree.
    Hierarchical clustering creates clusters within clusters, forming a tree-like structure.
        This structure is shown using a dendrogram (tree diagram).
        Dendogram:-
            At the bottom, every point is its own cluster.
            As you move up, clusters start merging.
            At the top, all points are in one big cluster.
    
    There are two types of approach for Hierarchical Clustering
    1. Agglomerative (Bottom-Up) — most common
        Start: Each point is its own cluster.                                          
        Step 1: Merge the two closest clusters.
        Step 2: Repeat until there’s one big cluster.
        Eg:-
            Step 0:   A    B    C    D   (4 separate clusters)
            Step 1:   (A,B)   C    D     (A and B merged first)
            Step 2:   (A,B)   (C,D)      (C and D merged)
            Step 3:   (A,B,C,D)          (all merged into one cluster)
    2. Divisive (Top-Down) — less common
        Start: All points in one big cluster.
        Step 1: Split it into smaller clusters.
        Step 2: Keep splitting until each point is alone.
        Eg:-
            Step 0:   (A,B,C,D)      (1 big cluster)
            Step 1:   (A,B)   (C,D)  (split into 2 smaller clusters)
            Step 2:   A   B   (C,D)  (split A,B apart)
            Step 3:   A   B   C   D  (split C,D apart → all separate)

    How Do We Decide Which Clusters to Merge?
    Why we need linkage
        At the very start:
        Every point is its own cluster.
        We can measure similarity (distance) between two points directly.
        But as soon as we merge two points into a cluster, we now have a set of points vs. another set of points.
        How do we measure the “distance” between two clusters?
        That’s where linkage criteria comes in.
        When merging clusters, we need a distance measure between clusters:
            Single Linkage – smallest distance between any two points in clusters.
            Complete Linkage – largest distance between any two points in clusters.
            Average Linkage – average distance between points in clusters.
            Ward’s Method – minimizes variance inside clusters (most popular).

    Key advantage
        You don’t need to know the number of clusters beforehand.
        You can “cut” the dendrogram at any height to get as many clusters as you want.


Now come the most IMPORTANT part that is coding 
Step 1 — Understand the Problem
    Before writing code, ask yourself:
        What input do I have? (dataset, number of clusters, distance metric)
        What output do I want? (cluster labels, dendrogram)
        Which algorithm type? (Agglomerative or Divisive)
    Example:
        Input: 6 points in 2D, want 2 clusters using Ward linkage
        Output: A list of cluster labels [0, 0, 1, 1, 0, 1]
Step 2 — Break the Algorithm into Steps
    For Agglomerative hierarchical clustering, the process is:
        1.Start with each point in its own cluster.
        → Data structure: a list of clusters, each being a list of point indices.
        2.Find the closest two clusters (using chosen distance metric).
        → Requires a distance function.
        3.Merge them into one cluster.
        → Modify your clusters list.
        4.Repeat until desired number of clusters is reached.
        5.Output cluster assignments.
    For Divisive 
        1.Start with all points in one cluster
        Data structure: [[0, 1, 2, 3, ...]]
        2.Pick the cluster to split
        Usually the largest cluster or the one with highest internal variance (most spread-out points).
        3.Split that cluster into two smaller clusters
        How to split?
            Run k-means with k=2
            Or find the point farthest apart and split based on distance.
        4.Replace the original cluster with the two new clusters
        Example: if [[0, 1, 2, 3]] splits into [[0, 1]] and [[2, 3]], update your cluster list.
        5.Repeat splitting until you have the desired number of clusters (k).
        6.Assign labels to each point based on which cluster it belongs to.
Step 3 — Think About Data Structures
    Points → store in a NumPy array (fast math operations).
    Clusters → list of lists of indices ([[0], [1], [2]...]).
    Distances → either recompute every time or store in a matrix.
Step 4 — Functions You’ll Need
    euclidean_distance(p1, p2)
        Purpose: Calculates the straight-line distance between two points in space.
        Used In: Both Agglomerative and Divisive clustering.
        Notes: Formula is √Σ((xᵢ − yᵢ)²); often the default choice in clustering due to its geometric meaning.
     manhattan_distance(p1, p2)
        Purpose: Calculates the sum of absolute coordinate differences between two points.
        Used In: Both Agglomerative and Divisive clustering.
        Notes: Formula is Σ|xᵢ − yᵢ|; better for grid-like movement or when outliers affect Euclidean distance too much.
    single_linkage_distance(cluster_a, cluster_b, X, dist_fn)
        Purpose: Finds the smallest distance between any point in one cluster and any point in another (Shortest Path method).
        Used In: Agglomerative clustering.
        Notes: Produces long, chain-like clusters because it merges based on the nearest pair of points.
    complete_linkage_distance(cluster_a, cluster_b, X, dist_fn)
        Purpose: Finds the largest distance between points in different clusters.
        Used In: Agglomerative clustering.
        Notes: Produces compact, spherical clusters but can break apart elongated shapes.
    average_linkage_distance(cluster_a, cluster_b, X, dist_fn)
        Purpose: Finds the mean of all point-to-point distances between two clusters.
        Used In: Agglomerative clustering.
        Notes: Balances between single and complete linkage; less sensitive to outliers.
    find_closest_clusters(clusters, X, linkage_fn, dist_fn)
        Purpose: Loops over all cluster pairs to find the two clusters with the smallest linkage distance.
        Used In: Agglomerative clustering.
        Notes: This is the core decision-making step for merges.
    agglomerative_clustering(X, k, linkage_fn, dist_fn)
        Purpose: Main loop that starts with each point as its own cluster and merges until k clusters remain.
        Used In: Agglomerative clustering.
        Notes: Relies heavily on linkage and distance functions for its behavior.
    find_farthest_points(cluster, X, dist_fn)
        Purpose: Finds the two most distant points inside a cluster.
        Used In: Divisive clustering.
        Notes: Used to determine the split “anchors” in distance-based divisive methods.
    split_cluster_by_distance(cluster, X, dist_fn)
        Purpose: Splits a cluster into two by assigning each point to the closer of the two farthest points.
        Used In: Divisive clustering.
        Notes: Simple but effective when combined with a variance-based cluster selection.
    divisive_clustering(X, k, dist_fn)
        Purpose: Main loop for Divisive clustering — starts with one cluster and repeatedly splits until k clusters are formed.
        Used In: Divisive clustering.
        Notes: Works best with distance-based splitting or k-means splitting.
    get_distance_function(name)
        Purpose: Returns the correct distance function (euclidean or manhattan) based on user input.
        Used In: Both Agglomerative and Divisive clustering.
        Notes: Helps make the code modular and easy to configure.
    get_linkage_function(name)
        Purpose: Returns the correct linkage function (single, complete, average) for Agglomerative clustering.
        Used In: Agglomerative clustering.
        Notes: Lets you change clustering behavior without touching the core algorithm.
Step 5 — Write Pseudocode
    Pseudocode forces you to think logically before coding.
          ->function hierarchical_clustering(data, k, method):
            make each point its own cluster
            while number_of_clusters > k:
                find the two closest clusters
                merge them
            return cluster assignments
Step 6 — Write the Code in Small Pieces
    1.Don’t write everything at once.
    2.First, make a distance function and test it.
    3.Then, try making your initial clusters.
    4.Then, merge two clusters manually to see if logic works.
    5.Finally, put it into a loop.
Step 7 — Test with Small Data
    Test with something tiny like:
        X = [[1, 2], [2, 3], [10, 10]]
    You can easily see if the merges make sense.

HERE BASICS FOR HIERARCHICAL CLUSTERING IS DONE

What’s a Dendrogram?
    A dendrogram is a tree-like diagram that shows:
        How points/clusters are merged at each step in hierarchical clustering.
        The vertical axis (y-axis) → the distance (or dissimilarity) at which the merge happened.
        The horizontal axis (x-axis) → the points (or clusters), usually in their original order.
    How to Read It
        Bottom (y=0) → Each leaf is an original data point.
        Moving upward → Clusters merge.
        Height of merge → Distance between the merged clusters.
        Higher merges → The clusters were more dissimilar.
    Cutting the Tree
        “Cutting the dendrogram” means drawing a horizontal line across it at some height:
            All merges below this line are kept as separate clusters.
            All merges above are ignored for that cut.
        Rules of Thumb:
            Lower cut → more clusters (fine-grained).
            Higher cut → fewer clusters (more general).

✅ Advantages
    No need to pre-specify 𝑘
        Unlike k-means, you don’t have to decide the number of clusters beforehand.
        You can choose later by cutting the dendrogram at the desired level.
    Visual representation via dendrograms
        Makes it easy to understand the merge/split process and explore different cluster counts.
    Flexible distance metrics
        Works with Euclidean, Manhattan, cosine similarity, and even custom distance functions.
    Captures nested structure
        Can detect smaller clusters within larger ones due to the hierarchical nature.
❌ Disadvantages
    Computationally expensive for large datasets
        Time complexity is typically O(n^3), space O(n^2), which is slow for thousands of points.
    Sensitive to noise/outliers
        Outliers can cause misleading merges and distort the hierarchy.
    Difficult to correct mistakes
        Once clusters are merged or split, the process cannot be undone — errors in early stages propagate.
    
When to Use Which Distance & Linkage
🔹 Linkage Methods
    Single Linkage –
        Best for long, chain-like clusters.
        Advantage: Can detect irregular or elongated shapes.
        Disadvantage: Prone to chaining effect (can connect distant points if intermediate points exist).
    Complete Linkage –
        Best for compact, spherical clusters.
        Advantage: Produces small, tight clusters.
        Disadvantage: May break apart naturally long shapes.
    Average Linkage –
        Balanced approach between Single & Complete linkage.
        Advantage: Works well in mixed situations.
        Disadvantage: Still sensitive to noise in data.
🔹 Distance Metrics
    Euclidean Distance –
        Use when geometric closeness is important.
        Straight-line distance formula between two points.
    Manhattan Distance –
        Use for grid-based or high-dimensional data.
        Measures distance along axes (sum of absolute differences).
    Cosine Distance –
        Use for text similarity or direction-based problems.
        Based on the angle between two vectors.
    Hamming Distance –
        Use for binary or categorical data.
        Counts the number of differing attributes.

Handling Large Datasets in Hierarchical Clustering
    1.Performance Limitation
        Hierarchical clustering has O(n³) time complexity in its naïve form.
        Memory usage is also high because it stores a distance matrix of size n × n.
    2.For Big n (large datasets):
        Sampling – Use a smaller subset of the data to build the hierarchy, then assign remaining points later.
        Approximations – Use algorithms like BIRCH or CURE that approximate hierarchical clustering.
        Hybrid Approach –
            First, cluster the dataset with a fast algorithm like K-Means.
            Then, run hierarchical clustering on the resulting centroids.
    3.When to Switch to Other Methods:
        Dataset size exceeds a few thousand points.
        Real-time or low-latency clustering is required.
        Memory constraints prevent storing the full distance matrix.


ALL CODE EXPLANATION 
1.AGGLOMATIVE 
    A.COMPLETE LINKAGE (EUCLIDEAN DISTANCE)
        Libraries Used
            numpy: To store and handle numerical data.
            scipy.cluster.hierarchy: For hierarchical clustering functions (linkage, dendrogram, fcluster).
            matplotlib.pyplot: For plotting dendrograms and scatter plots.
        Dataset Creation
            Created a numpy array X with 6 two-dimensional points.
            Points P1–P3 form one close group, and P4–P6 form another.
        Perform Clustering
            Used linkage(X, method='complete', metric='euclidean').
            Complete linkage: Distance between two clusters = maximum pairwise distance between points in those clusters.
            Euclidean metric: Straight-line distance.
        Plot Dendrogram
            Used dendrogram() to visualize the hierarchical merging of points/clusters.
            Labels assigned as P1 to P6.
            Y-axis shows merge distance.
        Form Flat Clusters
            Used fcluster(Z, t=2, criterion='maxclust') to cut the dendrogram into 2 clusters.
            Returns cluster assignments (e.g., [1 1 1 2 2 2]).
        Visualize Clusters on Original Data
            Used plt.scatter() to plot the original data points.
            Colored points by cluster ID (c=clusters, cmap='viridis').
            Added point labels (P1 to P6).
            Added grid and color bar for clarity.
        Result
            Dendrogram: Shows hierarchical relationships between points.
            Scatter Plot: Shows original data points colored by their cluster.
    B. SINGLE LINKAGE (EUCLIDEAN DISTANCE)



            
